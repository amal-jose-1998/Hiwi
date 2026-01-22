"""
Proper evaluation for held-out geometries in an auto-decoder SDF setup.

- Load trained decoder weights from a Lightning checkpoint (.ckpt) or a raw state_dict/.pt.
- For each TEST geometry:
    * load its SDF samples (points, sdf) from tool_geom{gid}_sdf.npz
    * apply the same normalization (stats from training)
    * freeze decoder
    * optimize a fresh latent vector z_g to fit that geometry (clamped L1 + lambda||z||^2)
    * report fit/eval loss on held-out points for that geometry
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .config import Network1Config
from .model import Network1SDF
#from .data import load_normalisation_stats, SDFTransform


def build_model_from_ckpt(ckpt, device):
    cfg = Network1Config()

    # get state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

    latent_dim, hidden_dim, depth, use_skip, skip_layer, activation, latent_init_std = infer_arch_from_state_dict(
        state, cfg_fallback=cfg
    )

    model = Network1SDF(
        num_geometries=1,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        latent_init_std=latent_init_std,
        activation=activation,
        use_skip=use_skip,
        skip_layer=skip_layer,
    ).to(device)

    return model, latent_dim, state


def load_decoder_only(model, ckpt_path):
    """
    Load only the decoder weights (fcs + out) from either:
      - Lightning .ckpt  (expects key: "state_dict", with "model.fcs.*" and "model.out.*")
      - plain torch .pt  (expects key: "model_state" or raw dict)
    We intentionally ignore latent embedding weights at eval (fresh z is optimized).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Pick the right state dict field
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]          # Lightning
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]         # optional legacy format
    elif isinstance(ckpt, dict):
        state = ckpt                        # raw state_dict
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Keep ONLY decoder weights and map keys if needed
    decoder_state = {}
    for k, v in state.items():
        # Lightning keys
        if k.startswith("model.fcs.") or k.startswith("model.out."):
            decoder_state[k[len("model."):]] = v  # => "fcs.*" or "out.*"
        # Raw model keys
        elif k.startswith("fcs.") or k.startswith("out."):
            decoder_state[k] = v

    if not decoder_state:
        some_keys = list(state.keys())[:50]
        raise RuntimeError(
            "Could not find any decoder keys (fcs/out) in checkpoint. "
            f"First keys: {some_keys}"
        )

    missing, unexpected = model.load_state_dict(decoder_state, strict=False)

    print(f"[eval] Loaded decoder (fcs+out only) from {ckpt_path}")
    if missing:
        print(f"[eval] Missing keys (expected: latent.*): {missing}")
    if unexpected:
        print(f"[eval] Unexpected keys (should be empty): {unexpected}")

@torch.no_grad()
def clamped_l1(pred, target, delta):
    if delta is None:
        return torch.mean(torch.abs(pred - target))
    pred_c = torch.clamp(pred, -delta, delta)
    tgt_c = torch.clamp(target, -delta, delta)
    return torch.mean(torch.abs(pred_c - tgt_c))


@torch.no_grad()
def eval_loss(model, z, xyz, sdf, delta):
    pred = model.forward_with_latent(z, xyz)
    return float(clamped_l1(pred, sdf, delta=delta).item())


def infer_arch_from_state_dict(state, cfg_fallback):
    """
    Infer latent_dim, hidden_dim, depth, and skip_layer from state_dict shapes.

    Works for Lightning keys ("model.fcs.*") or raw keys ("fcs.*").
    """
    # collect fc weight keys in order
    def norm_key(k):
        return k.replace("model.", "")  # strip if present

    fc_w = []
    for k, v in state.items():
        nk = norm_key(k)
        if nk.startswith("fcs.") and nk.endswith(".weight"):
            # k like fcs.0.weight
            idx = int(nk.split(".")[1])
            fc_w.append((idx, v.shape))

    if not fc_w:
        raise RuntimeError("Could not find any fcs.*.weight in checkpoint.")

    fc_w.sort(key=lambda x: x[0])

    # infer dims
    hidden_dim = int(fc_w[0][1][0])                  # out dim of first FC
    base_in_dim = int(fc_w[0][1][1])                 # in dim of first FC = latent_dim + 3
    latent_dim = base_in_dim - 3

    num_hidden = len(fc_w)                           # number of fcs layers
    depth = num_hidden + 1                           # + out layer

    # infer skip_layer: find which fc layer has in_dim = hidden_dim + base_in_dim
    skip_layer = None
    for idx, shape in fc_w:
        in_dim = int(shape[1])
        if in_dim == hidden_dim + base_in_dim:
            # this layer consumes concatenated [h, inp], so skip was applied after previous layer
            skip_layer = idx - 1
            break

    use_skip = skip_layer is not None

    # sanity print
    print(f"[eval] Inferred from ckpt: latent_dim={latent_dim}, hidden_dim={hidden_dim}, depth={depth}, "
          f"use_skip={use_skip}, skip_layer={skip_layer}")

    # activation cannot be inferred reliably from weights; use cfg fallback
    activation = getattr(cfg_fallback, "activation", "relu")
    latent_init_std = float(getattr(cfg_fallback, "latent_init_std", 0.012))

    return latent_dim, hidden_dim, depth, use_skip, skip_layer, activation, latent_init_std


def optimize_latent_for_geom( model, xyz_fit, sdf_fit, latent_dim, latent_init_std, latent_lr, latent_steps, latent_l2_weight, delta, device):
    """
    Optimize a fresh latent vector z for one geometry with decoder frozen.
    Uses the same (clamped) reconstruction loss as training for consistency.
    """
    z = torch.randn(latent_dim, device=device) * float(latent_init_std)
    z = torch.nn.Parameter(z)

    opt = torch.optim.Adam([z], lr=float(latent_lr))

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # freeze decoder

    for _ in range(int(latent_steps)):
        pred = model.forward_with_latent(z, xyz_fit)

        sdf_l1 = clamped_l1(pred, sdf_fit, delta=delta)
        reg = torch.mean(z ** 2)
        loss = sdf_l1 + float(latent_l2_weight) * reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return z.detach()


def main():
    cfg = Network1Config()
    
    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[eval] Device: {device}")

    ckpt_path = Path(cfg.eval_ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    rng = np.random.default_rng(int(cfg.eval_seed))

    # Read split table to get test geometry IDs
    geom_table = pd.read_parquet(cfg.geom_table_path)
    test_gids = geom_table.loc[geom_table["split"] == "test", "geometry_id"].astype(int).tolist()
    if len(test_gids) == 0:
        raise RuntimeError("No test geometries found in geom_table split.")

    print(f"[eval] #test_geometries={len(test_gids)} -> {test_gids}")

    # Load training normalization stats (computed on train geometries)
    #stats = load_normalisation_stats(cfg.stats_path)
    #sdf_scale = float(stats["sdf_scale"]) if float(stats["sdf_scale"]) != 0 else 1.0

    #transform = SDFTransform(
    #    xyz_mean=stats["xyz_mean"],
    #    xyz_std=stats["xyz_std"],
    #    sdf_scale=sdf_scale,
    #)

    if cfg.eval_loss_mode == "clamped":
        delta = float(getattr(cfg, "truncation_delta", 0.05))
        #delta = delta / max(sdf_scale, 1e-12)
        print(f"[eval] Using clamped loss: delta={delta}")
    else:
        delta = None
        print("[eval] Using unclamped L1 loss (not directly comparable to training clamped loss).")

    # Build a model with dummy embedding size; we will optimize z explicitly at test time
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model, latent_dim_ckpt, _ = build_model_from_ckpt(ckpt, device=device)
    load_decoder_only(model, ckpt_path)

    results = []
    sdf_dir = Path(cfg.sdf_dir)

    fit_frac = float(cfg.eval_fit_frac)
    if not (0.0 < fit_frac < 1.0):
        raise ValueError(f"cfg.eval_fit_frac must be in (0,1). Got: {fit_frac}")

    for gid in test_gids:
        npz_path = sdf_dir / f"tool_geom{gid}_sdf.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing SDF file: {npz_path}")

        data = np.load(npz_path)
        xyz = data["points"].astype(np.float32)
        sdf = data["sdf"].astype(np.float32)[:, None]

        # Apply same normalization used in training
        #xyz, sdf = transform(xyz, sdf)

        # Split into fit/eval subsets
        N = xyz.shape[0]
        perm = rng.permutation(N)
        n_fit = int(np.round(fit_frac * N))
        n_fit = max(1, min(n_fit, N - 1))
        fit_idx = perm[:n_fit]
        eval_idx = perm[n_fit:]

        xyz_fit = torch.from_numpy(xyz[fit_idx]).to(device)
        sdf_fit = torch.from_numpy(sdf[fit_idx]).to(device)
        xyz_eval = torch.from_numpy(xyz[eval_idx]).to(device)
        sdf_eval = torch.from_numpy(sdf[eval_idx]).to(device)

        # Optimize latent
        z = optimize_latent_for_geom(
            model=model,
            xyz_fit=xyz_fit,
            sdf_fit=sdf_fit,
            latent_dim=latent_dim_ckpt,
            latent_init_std=float(cfg.eval_latent_init_std),
            latent_lr=float(cfg.eval_latent_lr),
            latent_steps=int(cfg.eval_latent_steps),
            latent_l2_weight=float(cfg.latent_l2_weight),
            delta=delta,
            device=device,
        )

        # Evaluate
        fit_l1 = eval_loss(model, z, xyz_fit, sdf_fit, delta=delta)
        eval_l1 = eval_loss(model, z, xyz_eval, sdf_eval, delta=delta)

        results.append({
            "geometry_id": int(gid),
            "n_points": int(N),
            "n_fit": int(n_fit),
            "n_eval": int(N - n_fit),
            "fit_l1": float(fit_l1),
            "eval_l1": float(eval_l1),
        })

        print(f"[eval] gid={gid} fit_l1={fit_l1:.6f} eval_l1={eval_l1:.6f}")

    df = pd.DataFrame(results).sort_values("geometry_id").reset_index(drop=True)
    out_path = Path(cfg.eval_out_csv)
    df.to_csv(out_path, index=False)

    print(f"[eval] Saved results to {out_path}")
    print(f"[eval] Mean eval_l1 = {df['eval_l1'].mean():.6f}")


if __name__ == "__main__":
    main()
