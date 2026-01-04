"""
Proper evaluation for held-out geometries in an auto-decoder SDF setup.

Protocol:
- Load trained decoder weights from a Lightning checkpoint (.ckpt) or a raw state_dict/.pt.
- For each TEST geometry:
    * load its SDF samples (points, sdf) from tool_geom{gid}_sdf.npz
    * apply the same normalization (stats from training)
    * freeze decoder
    * optimize a fresh latent vector z_g to fit that geometry (clamped L1 + lambda||z||^2)
    * report fit/eval loss on held-out points for that geometry

This is the correct "test" for geometry-level split in an auto-decoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Network1Config
from .model import Network1SDF
from .data import load_normalisation_stats, SDFTransform


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
            decoder_state[k] = v
        # Raw model keys
        elif k.startswith("fcs.") or k.startswith("out."):
            decoder_state["model." + k] = v

    if not decoder_state:
        some_keys = list(state.keys())[:50]
        raise RuntimeError(
            "Could not find any decoder keys (fcs/out) in checkpoint. "
            f"First keys: {some_keys}"
        )

    missing, unexpected = model.load_state_dict(decoder_state, strict=False)

    print(f"[eval] Loaded decoder (fcs+out only) from {ckpt_path}")
    if unexpected:
        print(f"[eval] Unexpected keys: {unexpected}")
    # missing will include latent.* which is expected


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to network1 checkpoint (.ckpt/.pt)")
    parser.add_argument("--fit_frac", type=float, default=0.5, help="Fraction of points used to fit z (rest for eval)")
    parser.add_argument("--latent_steps", type=int, default=800, help="Optimization steps for each test geometry latent")
    parser.add_argument("--latent_lr", type=float, default=1e-2, help="LR for latent optimization")
    parser.add_argument("--latent_init_std", type=float, default=0.012, help="Init std for z (match training by default)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="network1_test_latent_opt.csv")
    parser.add_argument(
        "--use_clamped_loss",
        action="store_true",
        help="If set, evaluate and fit using the same clamped loss as training.",
    )
    args = parser.parse_args()

    cfg = Network1Config()
    rng = np.random.default_rng(int(args.seed))

    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[eval] Device: {device}")

    # Read split table to get test geometry IDs
    geom_table = pd.read_parquet(cfg.geom_table_path)
    test_gids = geom_table.loc[geom_table["split"] == "test", "geometry_id"].astype(int).tolist()
    if len(test_gids) == 0:
        raise RuntimeError("No test geometries found in geom_table split.")

    print(f"[eval] #test_geometries={len(test_gids)} -> {test_gids}")

    # Load training normalization stats (computed on train geometries)
    stats = load_normalisation_stats(cfg.stats_path)
    sdf_scale = float(stats["sdf_scale"]) if float(stats["sdf_scale"]) != 0 else 1.0

    transform = SDFTransform(
        xyz_mean=stats["xyz_mean"],
        xyz_std=stats["xyz_std"],
        sdf_scale=sdf_scale,
    )

    # Delta in normalized SDF units (because dataset divides sdf by sdf_scale)
    if args.use_clamped_loss:
        delta_phys = float(getattr(cfg, "truncation_delta", 0.05))
        delta_norm = delta_phys / max(sdf_scale, 1e-12)
        print(f"[eval] Using clamped loss: delta_phys={delta_phys}  sdf_scale={sdf_scale}  delta_norm={delta_norm}")
    else:
        delta_norm = None
        print("[eval] Using unclamped L1 loss (not directly comparable to training clamped loss).")

    # Build a model with dummy embedding size; we will optimize z explicitly at test time
    model = Network1SDF(
        num_geometries=1,  # dummy
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        latent_init_std=float(getattr(cfg, "latent_init_std", 0.012)),
        activation=getattr(cfg, "activation", "relu"),
        use_skip=bool(getattr(cfg, "use_skip", True)),
        skip_layer=getattr(cfg, "skip_layer", None),
    ).to(device)

    load_decoder_only(model, Path(args.ckpt))

    results = []
    sdf_dir = Path(cfg.sdf_dir)

    for gid in test_gids:
        npz_path = sdf_dir / f"tool_geom{gid}_sdf.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing SDF file: {npz_path}")

        data = np.load(npz_path)
        xyz = data["points"].astype(np.float32)
        sdf = data["sdf"].astype(np.float32)[:, None]

        # Apply same normalization used in training
        xyz_n, sdf_n = transform(xyz, sdf)

        # Split into fit/eval subsets
        N = xyz_n.shape[0]
        perm = rng.permutation(N)
        n_fit = int(np.round(float(args.fit_frac) * N))
        n_fit = max(1, min(n_fit, N - 1))
        fit_idx = perm[:n_fit]
        eval_idx = perm[n_fit:]

        xyz_fit = torch.from_numpy(xyz_n[fit_idx]).to(device)
        sdf_fit = torch.from_numpy(sdf_n[fit_idx]).to(device)
        xyz_eval = torch.from_numpy(xyz_n[eval_idx]).to(device)
        sdf_eval = torch.from_numpy(sdf_n[eval_idx]).to(device)

        # Optimize latent
        z = optimize_latent_for_geom(
            model=model,
            xyz_fit=xyz_fit,
            sdf_fit=sdf_fit,
            latent_dim=cfg.latent_dim,
            latent_init_std=float(args.latent_init_std),
            latent_lr=float(args.latent_lr),
            latent_steps=int(args.latent_steps),
            latent_l2_weight=float(cfg.latent_l2_weight),
            delta=delta_norm,
            device=device,
        )

        # Evaluate
        fit_l1 = eval_loss(model, z, xyz_fit, sdf_fit, delta=delta_norm)
        eval_l1 = eval_loss(model, z, xyz_eval, sdf_eval, delta=delta_norm)

        results.append({
            "geometry_id"(gid),
            "n_points"(N),
            "n_fit"(n_fit),
            "n_eval"(N - n_fit),
            "fit_l1"(fit_l1),
            "eval_l1"(eval_l1),
        })

        print(f"[eval] gid={gid} fit_l1={fit_l1:.6f} eval_l1={eval_l1:.6f}")

    df = pd.DataFrame(results).sort_values("geometry_id").reset_index(drop=True)
    out_path = Path(args.out_csv)
    df.to_csv(out_path, index=False)

    print(f"[eval] Saved results to {out_path}")
    print(f"[eval] Mean eval_l1 = {df['eval_l1'].mean():.6f}")


if __name__ == "__main__":
    main()
