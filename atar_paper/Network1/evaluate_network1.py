"""
Proper evaluation for held-out geometries in an auto-decoder SDF setup.

Protocol:
- Load trained decoder weights (MLP) from a checkpoint.
- For each TEST geometry:
    * load its SDF samples (points, sdf) from tool_geom{gid}_sdf.npz
    * apply the same normalization (stats from training)
    * freeze decoder
    * optimize a fresh latent vector z_g to fit that geometry (L1 + lambda||z||^2)
    * report eval loss on held-out points for that geometry

This is the correct "test" for geometry-level split.
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


def load_decoder_only(model: Network1SDF, ckpt_path: Path) -> None:
    """
    Load only the decoder (MLP) weights from either:
      - Lightning .ckpt  (expects key: "state_dict", with "model.mlp.*")
      - plain torch .pt  (expects key: "model_state", with "mlp.*" or "model.mlp.*")
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 1) pick the right state dict field
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]          # Lightning
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]         # your old manual checkpoints
    elif isinstance(ckpt, dict):
        # sometimes people save raw state_dict directly
        state = ckpt
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

    # 2) keep ONLY the decoder weights and map keys to model.mlp.*
    mlp_state = {}
    for k, v in state.items():
        if k.startswith("model.mlp."):
            mlp_state[k] = v
        elif k.startswith("mlp."):
            mlp_state["model." + k] = v  # turn "mlp.0.weight" -> "model.mlp.0.weight"

    if not mlp_state:
        # helpful debug print
        some_keys = list(state.keys())[:30]
        raise RuntimeError(
            "Could not find any MLP keys in checkpoint. "
            f"First keys: {some_keys}"
        )

    missing, unexpected = model.load_state_dict(mlp_state, strict=False)

    print(f"[eval] Loaded decoder (MLP only) from {ckpt_path}")
    if unexpected:
        print(f"[eval] Unexpected keys: {unexpected}")
    # missing will include: "latent.*" which is expected (we ignore embedding at eval)



@torch.no_grad()
def eval_loss(model: Network1SDF, z: torch.Tensor, xyz: torch.Tensor, sdf: torch.Tensor) -> float:
    """Compute mean L1 loss for a fixed latent z on a set of points."""
    B = xyz.shape[0]
    zB = z.view(1, -1).expand(B, -1)
    pred = model.mlp(torch.cat([zB, xyz], dim=1))
    return float(torch.mean(torch.abs(pred - sdf)).item())


def optimize_latent_for_geom(
    model: Network1SDF,
    xyz_fit: torch.Tensor,
    sdf_fit: torch.Tensor,
    latent_dim: int,
    latent_init_std: float,
    latent_lr: float,
    latent_steps: int,
    latent_l2_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Optimize a fresh latent vector z for one geometry with decoder frozen.
    """
    # fresh latent for this geometry
    z = torch.randn(latent_dim, device=device) * float(latent_init_std)
    z = torch.nn.Parameter(z)

    opt = torch.optim.Adam([z], lr=float(latent_lr))

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # freeze decoder

    for _ in range(int(latent_steps)):
        B = xyz_fit.shape[0]
        zB = z.view(1, -1).expand(B, -1)
        pred = model.mlp(torch.cat([zB, xyz_fit], dim=1))

        sdf_l1 = torch.mean(torch.abs(pred - sdf_fit))
        reg = torch.mean(z ** 2)
        loss = sdf_l1 + float(latent_l2_weight) * reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return z.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to network1 checkpoint (.pt)")
    parser.add_argument("--fit_frac", type=float, default=0.5, help="Fraction of points used to fit z (rest for eval)")
    parser.add_argument("--latent_steps", type=int, default=800, help="Optimization steps for each test geometry latent")
    parser.add_argument("--latent_lr", type=float, default=1e-2, help="LR for latent optimization")
    parser.add_argument("--latent_init_std", type=float, default=0.01, help="Init std for z")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="network1_test_latent_opt.csv")
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
    transform = SDFTransform(
        xyz_mean=stats["xyz_mean"],
        xyz_std=stats["xyz_std"],
        sdf_scale=float(stats["sdf_scale"]),
    )

    # Build a model with dummy embedding size; we will only use model.mlp
    model = Network1SDF(
        num_geometries=1,  # dummy, not used in evaluation
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
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
            latent_init_std=args.latent_init_std,
            latent_lr=args.latent_lr,
            latent_steps=args.latent_steps,
            latent_l2_weight=cfg.latent_l2_weight,
            device=device,
        )

        # Evaluate
        fit_l1 = eval_loss(model, z, xyz_fit, sdf_fit)
        eval_l1 = eval_loss(model, z, xyz_eval, sdf_eval)

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
    out_path = Path(args.out_csv)
    df.to_csv(out_path, index=False)
    print(f"[eval] Saved results to {out_path}")
    print(f"[eval] Mean eval_l1 = {df['eval_l1'].mean():.6f}")


if __name__ == "__main__":
    main()
