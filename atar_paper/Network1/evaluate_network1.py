"""
Network-1 evaluation 

This script evaluates held-out geometries for an auto-decoder SDF model using
the standard latent-optimization protocol (DeepSDF-style):

For each TEST geometry and each tool component:
  1) load SDF samples tool_geom{gid}_{comp}_sdf.npz
  2) load decoder weights from cfg.eval_ckpt_path (fcs + out only)
  3) freeze decoder
  4) optimize a fresh latent vector z_{gid,comp} on a subset of points
  5) report fit / eval error (clamped L1 or unclamped L1 based on cfg.eval_loss_mode)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .config import Network1Config
from .model import Network1SDF


def _pick_device(cfg_device):
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_decoder_only(model, ckpt_path) -> None:
    """
    Load only the decoder weights (fcs + out) from either:
      - Lightning .ckpt  (expects key: "state_dict" with "model.fcs.*" and "model.out.*")
      - raw state_dict   (expects "fcs.*" and "out.*")
      - dict with "model_state" (optional legacy)

    Latent embedding weights are intentionally ignored at eval time.
    """
    # Loading the trained weights (excluding latents)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]  # Lightning
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

    mapped = {}
    for k, v in state.items():
        # strip lightning prefix
        if k.startswith("model."):
            k2 = k.replace("model.", "", 1)
        else:
            k2 = k

        # skip latent table entirely
        if k2.startswith("latent."):
            continue

        mapped[k2] = v

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"[eval] Loaded all non-latent weights from {ckpt_path}")
    if unexpected:
        print(f"[eval] Unexpected keys: {unexpected}")
    # missing will include latent.* which is expected


@torch.no_grad()
def l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


@torch.no_grad()
def clamped_l1_loss(pred, target, delta):
    pred_c = torch.clamp(pred, -delta, delta)
    tgt_c = torch.clamp(target, -delta, delta)
    return torch.mean(torch.abs(pred_c - tgt_c))


def _loss_fn(pred, target, loss_mode, delta):
    if loss_mode == "clamped":
        return clamped_l1_loss(pred, target, delta=delta)
    if loss_mode == "l1":
        return l1_loss(pred, target)
    raise ValueError(f"Unknown eval_loss_mode='{loss_mode}'. Expected 'clamped' or 'l1'.")


@torch.no_grad()
def eval_error(model, z, xyz, sdf, loss_mode, delta):
    # evaluates the decoder on a set of points, using a specific latent z.
    pred = model.forward_with_latent(z, xyz)
    return float(_loss_fn(pred, sdf, loss_mode=loss_mode, delta=delta).item())


def optimize_latent(model, xyz_fit, sdf_fit, latent_dim, latent_init_std, latent_lr,
    latent_steps, latent_l2_weight, loss_mode, delta, device):
    """
    Optimize a fresh latent vector z for a single (geometry, component).
    Decoder weights are frozen; only z is updated.
    """
    # Initialize a fresh latent z
    z = torch.randn(latent_dim, device=device) * float(latent_init_std)
    z = torch.nn.Parameter(z)

    opt = torch.optim.Adam([z], lr=float(latent_lr)) # Optimizer that updates only z

    # Freeze the decoder, so that gradients flow through the decoder into z, but decoder weights don’t change.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Iterative optimization loop
    for _ in range(int(latent_steps)):
        pred = model.forward_with_latent(z, xyz_fit)
        recon = _loss_fn(pred, sdf_fit, loss_mode=loss_mode, delta=delta) # match the SDF samples
        reg = torch.mean(z ** 2) # keep the latent from blowing up
        loss = recon + float(latent_l2_weight) * reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return z.detach() # the optimized latent vector for that test geometry/component.


def main():
    cfg = Network1Config()
    ckpt_path = Path(cfg.eval_ckpt_path)
    fit_frac = float(cfg.eval_fit_frac)
    latent_steps = int(cfg.eval_latent_steps)
    latent_lr = float(cfg.eval_latent_lr)
    latent_init_std = float(cfg.eval_latent_init_std)
    seed = int(cfg.eval_seed)
    out_csv = Path(cfg.eval_out_csv)
    loss_mode = str(cfg.eval_loss_mode)

    device = _pick_device(str(cfg.device))
    print(f"[eval] Device: {device}")

    # config-driven essentials
    tool_components = list(cfg.tool_components)
    delta = float(cfg.truncation_delta)

    print(f"[eval] ckpt={ckpt_path}")
    print(f"[eval] loss_mode={loss_mode} (delta={delta} if clamped)")
    print(f"[eval] fit_frac={fit_frac}, latent_steps={latent_steps}, latent_lr={latent_lr}, latent_init_std={latent_init_std}, seed={seed}")
    print(f"[eval] tool_components={tool_components}")

    if not (0.0 < fit_frac < 1.0):
        raise ValueError("fit_frac must be in (0,1)")

    rng = np.random.default_rng(seed)

    # Read test geometry IDs from the metadata table
    geom_table = pd.read_parquet(cfg.geom_table_path)
    test_gids = geom_table.loc[geom_table["split"] == "test", "geometry_id"].astype(int).tolist()

    if len(test_gids) == 0:
        raise RuntimeError("No test geometries found in geom_table split.")
    print(f"[eval] #test_geometries={len(test_gids)} -> {test_gids}")

    # Build a model shell; latents are optimized explicitly at test time.
    # Only fcs/out are needed; latent embedding is ignored.
    model = Network1SDF(
        num_geometries=1, # at eval time we are not using the embedding table at all.
        num_components=len(tool_components),
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        latent_init_std=float(cfg.latent_init_std),
        activation=str(cfg.activation),
        use_skip=bool(cfg.use_skip),
        skip_layer=cfg.skip_layer,
    ).to(device)

    load_decoder_only(model, ckpt_path) # Load trained non-latent weights

    sdf_dir = Path(cfg.sdf_dir)

    results = []
    # For each test geometry and each component
    for gid in test_gids:
        for comp in tool_components:
            npz_path = sdf_dir / f"tool_geom{gid}_{comp}_sdf.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(f"Missing SDF file: {npz_path}")

            data = np.load(npz_path, allow_pickle=True)
            xyz = data["points"].astype(np.float32)
            sdf = data["sdf"].astype(np.float32)[:, None]

            N = xyz.shape[0]
            # Randomly split points into “fit” and “eval”
            perm = rng.permutation(N)
            n_fit = int(np.round(fit_frac * N))
            n_fit = max(1, min(n_fit, N - 1))
            fit_idx = perm[:n_fit]
            eval_idx = perm[n_fit:]

            xyz_fit = torch.from_numpy(xyz[fit_idx]).to(device) # used to optimize latent z
            sdf_fit = torch.from_numpy(sdf[fit_idx]).to(device)

            xyz_eval = torch.from_numpy(xyz[eval_idx]).to(device) # never seen during optimization; used to test generalization within that geometry
            sdf_eval = torch.from_numpy(sdf[eval_idx]).to(device)

            # Optimize latent on fit points
            z = optimize_latent(
                model,
                xyz_fit,
                sdf_fit,
                latent_dim=int(cfg.latent_dim),
                latent_init_std=float(latent_init_std),
                latent_lr=float(latent_lr),
                latent_steps=int(latent_steps),
                latent_l2_weight=float(cfg.latent_l2_weight),
                loss_mode=loss_mode,
                delta=delta,
                device=device,
            )

            # Measure fit and eval error
            fit_err = eval_error(model, z, xyz_fit, sdf_fit, loss_mode=loss_mode, delta=delta) # should be low
            eval_err = eval_error(model, z, xyz_eval, sdf_eval, loss_mode=loss_mode, delta=delta) # tells us how well the learned decoder represents that geometry/component beyond the specific fit subset

            results.append(
                {
                    "geometry_id": int(gid),
                    "component": comp,
                    "n_points": int(N),
                    "n_fit": int(n_fit),
                    "n_eval": int(N - n_fit),
                    "fit_l1": float(fit_err),
                    "eval_l1": float(eval_err),
                }
            )

            print(f"[eval] gid={gid} comp={comp} fit_l1={fit_err:.6f} eval_l1={eval_err:.6f}")

    df = pd.DataFrame(results).sort_values(["geometry_id", "component"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[eval] Saved results to {out_csv}")
    print(f"[eval] Mean eval_l1 = {df['eval_l1'].mean():.6f}")

    print("[eval] Mean eval_l1 by component:")
    for comp, m in df.groupby("component")["eval_l1"].mean().sort_index().items():
        print(f"  - {comp}: {m:.6f}")


if __name__ == "__main__":
    main()
