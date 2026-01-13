# atar_paper/Network2/preprocessing/build_surrogate_inputs.py

"""
Build cached Network-2 surrogate inputs using a trained Network-1 model.

Per geometry_id:
1) obtain z:
   - from checkpoint latent table if present and geometry_id is covered
   - else optionally optimize z using that geometry's SDF samples (tool_geom{gid}_sdf.npz)
2) z -> mesh via Network-1 decoder + Marching Cubes
3) mesh -> depth maps (X and Z projections)
4) join parameters:
   - geom_params from geom_table (per geometry_id)
   - proc_params from meta at rep_ID (representative simulation)
5) save:
   out_dir/geom{gid}_surrogate_inputs.npz
   out_dir/manifest_surrogate_inputs.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from atar_paper.Network1.config import Network1Config
from atar_paper.Network1.model import Network1SDF

from atar_paper.Network2.config import (
    MeshFromLatentConfig,
    DepthProjectionConfig,
    BuildSurrogateInputsConfig,
)
from atar_paper.Network2.preprocessing.mesh_from_latent import mesh_from_latent
from atar_paper.Network2.preprocessing.depth_projection import project_mesh_to_depth_maps


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
def _load_state_dict_any(ckpt_path):
    """
    Load a torch checkpoint and return a mapping like a state_dict.

    Parameters
    ckpt_path :
        Path to a Lightning .ckpt or raw .pt.

    Returns
    state :
        dict mapping parameter names to tensors.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")


def load_decoder_only(model, ckpt_path):
    """
    Load only decoder weights (fcs + out) from a Network-1 checkpoint.

    Parameters
    model :
        Network1SDF instance.
    ckpt_path :
        Path to Network-1 checkpoint.

    Returns
    None

    Raises
    RuntimeError
        If decoder keys cannot be found in the checkpoint.
    """
    state = _load_state_dict_any(ckpt_path)

    decoder_state = {}
    for k, v in state.items():
        # Lightning keys
        if k.startswith("model.fcs.") or k.startswith("model.out."):
            decoder_state[k] = v
        # Raw model keys
        elif k.startswith("fcs.") or k.startswith("out."):
            decoder_state[k] = v

    if not decoder_state:
        raise RuntimeError("Could not find decoder weights (fcs/out) in checkpoint.")

    model.load_state_dict(decoder_state, strict=False)


def try_load_latent_table(ckpt_path):
    """
    Try to extract the latent embedding table from a Network-1 checkpoint.

    Parameters
    ckpt_path :
        Path to Network-1 checkpoint.

    Returns
    latent_weight :
        torch.Tensor of shape (K, L) if present; otherwise None.
    """
    state = _load_state_dict_any(ckpt_path)
    for key in ("model.latent.weight", "latent.weight"):
        if key in state:
            w = state[key]
            if torch.is_tensor(w) and w.ndim == 2:
                return w.detach().clone()
    return None


# -----------------------------------------------------------------------------
# Latent inference (fallback)
# -----------------------------------------------------------------------------
def _clamped_l1(pred, target, delta):
    """
    Mean absolute error with optional truncation.

    Parameters
    pred :
        torch.Tensor of shape (N,1) or (N,).
    target :
        torch.Tensor, same shape as pred.
    delta :
        float or None. If provided, clamp both pred and target to [-delta, delta].

    Returns
    loss :
        torch.Tensor scalar.
    """
    if delta is None:
        return torch.mean(torch.abs(pred - target))
    pred_c = torch.clamp(pred, -delta, delta)
    tgt_c = torch.clamp(target, -delta, delta)
    return torch.mean(torch.abs(pred_c - tgt_c))


def optimize_latent_for_geom(model, xyz_fit, sdf_fit, latent_dim, latent_init_std,
    latent_lr, latent_steps, latent_l2_weight, delta):
    """
    Infer a latent z for one geometry by optimizing z with a frozen decoder.

    Parameters
    model :
        Network1SDF with trained decoder weights loaded.
    xyz_fit :
        torch.Tensor of shape (N, 3). Query points.
    sdf_fit :
        torch.Tensor of shape (N, 1). Ground-truth SDF values.
    latent_dim :
        int. Latent dimension L.
    latent_init_std :
        float. Std for z0 initialization.
    latent_lr :
        float. Adam learning rate for z.
    latent_steps :
        int. Number of optimization iterations.
    latent_l2_weight :
        float. Weight for ||z||^2 penalty.
    delta :
        float or None. If set, use clamped L1 data term.

    Returns
    z :
        torch.Tensor of shape (L,), detached, on same device as xyz_fit.
    """
    device = xyz_fit.device
    z = torch.randn((int(latent_dim),), device=device) * float(latent_init_std)
    z.requires_grad_(True)

    opt = torch.optim.Adam([z], lr=float(latent_lr))

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for _ in range(int(latent_steps)):
        pred = model.forward_with_latent(z, xyz_fit)  # (N,1)
        sdf_l1 = _clamped_l1(pred, sdf_fit, delta=delta)
        reg = torch.mean(z ** 2)
        loss = sdf_l1 + float(latent_l2_weight) * reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return z.detach()


# -----------------------------------------------------------------------------
# Tabular helpers
# -----------------------------------------------------------------------------
def _row_to_vector(row, cols):
    """
    Convert a pandas row into a float32 vector using selected columns.

    Parameters
    row :
        pandas.Series.
    cols :
        list of str. Columns extracted in order.

    Returns
    vec :
        np.ndarray of shape (len(cols),), dtype float32.
    """
    if len(cols) == 0:
        return np.zeros((0,), dtype=np.float32)

    vals = []
    for c in cols:
        v = row[c]
        vals.append(float(v) if pd.notna(v) else 0.0)
    return np.asarray(vals, dtype=np.float32)


# -----------------------------------------------------------------------------
# Core builder
# -----------------------------------------------------------------------------
def build_surrogate_inputs(build_cfg=None, mesh_cfg=None, depth_cfg=None):
    """
    Build and save Network-2 surrogate inputs according to configuration.

    Parameters
    build_cfg :
        BuildSurrogateInputsConfig or None.
        If None, defaults are used.
    mesh_cfg :
        MeshFromLatentConfig or None.
        If None, defaults are used.
    depth_cfg :
        DepthProjectionConfig or None.
        If None, defaults are used.

    Returns
    manifest_df :
        pandas.DataFrame summarizing outputs (one row per geometry_id).
    """
    build_cfg = BuildSurrogateInputsConfig() if build_cfg is None else build_cfg
    mesh_cfg = MeshFromLatentConfig() if mesh_cfg is None else mesh_cfg
    depth_cfg = DepthProjectionConfig() if depth_cfg is None else depth_cfg

    out_dir = Path(build_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geom_table = pd.read_parquet(build_cfg.geom_table_path)
    meta = pd.read_parquet(build_cfg.meta_path)

    if "geometry_id" not in geom_table.columns:
        raise RuntimeError("geom_table.parquet must contain 'geometry_id'.")
    if "rep_ID" not in geom_table.columns:
        raise RuntimeError("geom_table.parquet must contain 'rep_ID'.")
    if "ID" not in meta.columns:
        raise RuntimeError("meta.parquet must contain 'ID'.")

    # Index meta by simulation ID (rep_ID points into this)
    meta_by_ID = meta.reset_index(drop=True).set_index("ID", drop=False)

    # --- enforce strict separation of inputs ---

    # Geometry params from geom_table only
    geom_cols = [c for c in build_cfg.geom_param_cols if c in geom_table.columns]
    missing_geom = [c for c in build_cfg.geom_param_cols if c not in geom_table.columns]
    if missing_geom:
        raise RuntimeError(f"Missing geometry columns in geom_table: {missing_geom}")

    # Process params from meta only (and ensure geometry cols do not leak)
    proc_cols = [c for c in build_cfg.proc_param_cols if c in meta.columns]
    missing_proc = [c for c in build_cfg.proc_param_cols if c not in meta.columns]
    if missing_proc:
        raise RuntimeError(f"Missing process columns in meta: {missing_proc}")

    proc_cols = [c for c in proc_cols if c not in set(build_cfg.proc_exclude_cols)]
    if len(proc_cols) == 0:
        raise RuntimeError("proc_param_cols became empty after exclusions. Check config.")

    # RNG for reproducibility
    rng = np.random.default_rng(int(build_cfg.seed))

    # Load Network-1 decoder
    cfg1 = Network1Config()
    device = torch.device(cfg1.device if (cfg1.device == "cuda" and torch.cuda.is_available()) else "cpu")

    model = Network1SDF(
        num_geometries=1,  # we will call forward_with_latent(z, xyz)
        latent_dim=cfg1.latent_dim,
        hidden_dim=cfg1.hidden_dim,
        depth=cfg1.depth,
        latent_init_std=float(getattr(cfg1, "latent_init_std", 0.012)),
        activation=getattr(cfg1, "activation", "relu"),
        use_skip=bool(getattr(cfg1, "use_skip", True)),
        skip_layer=getattr(cfg1, "skip_layer", None),
    ).to(device)

    load_decoder_only(model, build_cfg.network1_ckpt)
    model.eval()

    # Latent table (optional)
    latent_table = try_load_latent_table(build_cfg.network1_ckpt) if build_cfg.use_latent_table_if_available else None
    latent_K = int(latent_table.shape[0]) if latent_table is not None else 0

    # Clamp delta for latent fitting (if enabled)
    delta = float(getattr(cfg1, "truncation_delta", 0.05)) if build_cfg.use_clamped_loss else None
    latent_l2_weight = float(getattr(cfg1, "latent_l2_weight", 1e-4))

    rows = []
    for _, g in geom_table.iterrows():
        gid = int(g["geometry_id"])
        split = str(g["split"]) if "split" in geom_table.columns else ""
        rep_ID = int(g["rep_ID"])

        # --- processing params (meta at rep_ID) ---
        if rep_ID not in meta_by_ID.index:
            raise RuntimeError(f"rep_ID={rep_ID} not found in meta.parquet 'ID' column.")
        meta_row = meta_by_ID.loc[rep_ID]
        if isinstance(meta_row, pd.DataFrame):
            meta_row = meta_row.iloc[0]
        proc_vec = _row_to_vector(meta_row, proc_cols)

        # --- geometry params (geom_table row) ---
        geom_vec = _row_to_vector(g, geom_cols)

        # --- latent z ---
        z_src = "none"
        if latent_table is not None and gid < latent_K:
            z = latent_table[gid].to(device=device, dtype=torch.float32).clone()
            z_src = "latent_table"
        else:
            if not build_cfg.infer_latent_if_missing:
                raise RuntimeError(
                    f"geometry_id={gid} not in latent table and infer_latent_if_missing=False."
                )

            npz_path = Path(build_cfg.sdf_dir) / f"tool_geom{gid}_sdf.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(f"Missing SDF file for latent inference: {npz_path}")

            data = np.load(npz_path)
            xyz = data["points"].astype(np.float32)
            sdf = data["sdf"].astype(np.float32)[:, None]

            N = xyz.shape[0]
            perm = rng.permutation(N)
            n_fit = int(np.round(float(build_cfg.fit_frac) * N))
            n_fit = max(1, min(n_fit, N))
            fit_idx = perm[:n_fit]

            xyz_fit = torch.from_numpy(xyz[fit_idx]).to(device=device)
            sdf_fit = torch.from_numpy(sdf[fit_idx]).to(device=device)

            z = optimize_latent_for_geom(
                model=model,
                xyz_fit=xyz_fit,
                sdf_fit=sdf_fit,
                latent_dim=cfg1.latent_dim,
                latent_init_std=float(build_cfg.latent_init_std),
                latent_lr=float(build_cfg.latent_lr),
                latent_steps=int(build_cfg.latent_steps),
                latent_l2_weight=latent_l2_weight,
                delta=delta,
            )
            z_src = "optimized"

        # --- z -> mesh -> depth maps ---
        V, F, mesh_meta = mesh_from_latent(model, z, cfg=mesh_cfg)
        depth_x, depth_z, mask_x, mask_z, proj_meta = project_mesh_to_depth_maps(
            V, F, cfg=depth_cfg, seed=int(build_cfg.seed)
        )

        # --- save per geometry ---
        out_path = out_dir / f"geom{gid}_surrogate_inputs.npz"
        np.savez_compressed(
            out_path,
            geometry_id=np.asarray([gid], dtype=np.int64),
            rep_ID=np.asarray([rep_ID], dtype=np.int64),
            split=np.asarray([split], dtype=object),

            depth_x=depth_x.astype(np.float32),
            depth_z=depth_z.astype(np.float32),
            mask_x=mask_x.astype(np.uint8),
            mask_z=mask_z.astype(np.uint8),

            geom_params=geom_vec.astype(np.float32),
            geom_param_names=np.asarray(geom_cols, dtype=object),

            proc_params=proc_vec.astype(np.float32),
            proc_param_names=np.asarray(proc_cols, dtype=object),

            z=z.detach().cpu().numpy().astype(np.float32),
            z_source=np.asarray([z_src], dtype=object),
        )

        rows.append(
            {
                "geometry_id": gid,
                "split": split,
                "rep_ID": rep_ID,
                "z_source": z_src,
                "out_npz": str(out_path),
                "n_verts": int(mesh_meta.get("n_verts", -1)),
                "n_faces": int(mesh_meta.get("n_faces", -1)),
                "depth_res": int(depth_cfg.resolution),
                "geom_dim": int(geom_vec.shape[0]),
                "proc_dim": int(proc_vec.shape[0]),
            }
        )

        print(f"[build_surrogate_inputs] gid={gid} split={split} z={z_src} -> {out_path.name}")

    manifest_df = pd.DataFrame(rows).sort_values("geometry_id").reset_index(drop=True)
    manifest_df.to_csv(out_dir / "manifest_surrogate_inputs.csv", index=False)

    return manifest_df


def run_build_surrogate_inputs():
    """
    Convenience runner using default configs.

    Returns
    manifest_df :
        pandas.DataFrame manifest of written files.
    """
    return build_surrogate_inputs(
        build_cfg=BuildSurrogateInputsConfig(),
        mesh_cfg=MeshFromLatentConfig(),
        depth_cfg=DepthProjectionConfig(),
    )
