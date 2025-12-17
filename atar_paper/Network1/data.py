"""Dataset and dataloader utilities for Network-1 (Auto-Decoder SDF)."""

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class ToolSDFDataset(Dataset):
    """
    Dataset for Network-1 (Auto-Decoder SDF).
    - The dataset provides (geom_id, xyz, sdf)
    - Latent vectors z_i are NOT part of the dataset
    - Latent vectors live inside the model as nn.Embedding, indexed by geom_id

    Each sample:
        geom_id : torch.long scalar in [0, K-1]
            Index of geometry (used to look up latent vector z_i in nn.Embedding).
        xyz : torch.float32 tensor, shape (3,)
            Query point in 3D space.
        sdf : torch.float32 tensor, shape (1,)
            Ground-truth signed distance at xyz.
    """

    def __init__(self, sdf_dir, transform=None):
        self.sdf_dir = Path(sdf_dir)
        self.sdf_files = sorted(self.sdf_dir.glob("tool_geom*_sdf.npz"))
        if not self.sdf_files:
            raise RuntimeError(f"No SDF npz files found in {self.sdf_dir}")

        self.transform = transform

        # Load and stack all samples
        xyz_all = []
        sdf_all = []
        geom_id_all = []

        for path in self.sdf_files:
            data = np.load(path)
            # N = number of SDF points per geometry
            points = data["points"].astype(np.float32)          # (N,3)
            sdf = data["sdf"].astype(np.float32)[:, None]       # (N,1)
            geom_id = int(data["geom_id"])

            N = points.shape[0]
            geom_ids = np.full((N,), geom_id, dtype=np.int64)   # (N,)

            xyz_all.append(points)
            sdf_all.append(sdf)
            geom_id_all.append(geom_ids)
        # M = total number of SDF points across all geometries => training samples
        self.xyz = np.vstack(xyz_all)                    # (M,3)
        self.sdf = np.vstack(sdf_all)                    # (M,1)
        self.geom_id = np.concatenate(geom_id_all)       # (M,)

        self.num_geometries = int(self.geom_id.max()) + 1

    def __len__(self):
        return int(self.xyz.shape[0])

    def __getitem__(self, idx):
        geom_id = int(self.geom_id[idx])
        xyz = self.xyz[idx]
        sdf = self.sdf[idx]

        if self.transform:
            xyz, sdf = self.transform(xyz, sdf)

        return (
            torch.tensor(geom_id, dtype=torch.long),
            torch.from_numpy(xyz).float(),
            torch.from_numpy(sdf).float(),
        )


class SDFTransform:
    """
    Normalisation transform for SDF training.

    - Normalises xyz using mean/std
    - Scales sdf by sdf_scale (max(|sdf|) from training set)
    """

    def __init__(self, xyz_mean, xyz_std, sdf_scale):
        self.xyz_mean = xyz_mean.astype(np.float32)
        self.xyz_std = np.where(xyz_std == 0, 1.0, xyz_std).astype(np.float32)
        self.sdf_scale = float(sdf_scale) if float(sdf_scale) != 0 else 1.0

    def __call__(self, xyz, sdf):
        xyz_norm = (xyz - self.xyz_mean) / self.xyz_std
        sdf_norm = sdf / self.sdf_scale
        return xyz_norm.astype(np.float32), sdf_norm.astype(np.float32)


class BalancedGeomBatchSampler(Sampler[List[int]]):
    """
    Balanced batch sampler: equal number of samples per geometry in each batch.

    Effective batch size:
        batch_size = num_geometries * samples_per_geom

    Requirements
    dataset.geom_id : np.ndarray of shape (M,)
        Geometry id per sample.
    """

    def __init__(self, dataset, samples_per_geom, seed=0):
        self.dataset = dataset
        self.samples_per_geom = int(samples_per_geom)
        self.seed = int(seed)
        self.epoch = 0

        geom_ids = dataset.geom_id
        self.unique_gids = np.unique(geom_ids).astype(np.int64).tolist()
        self.idx_by_gid = {                                           # key = geometry id
            gid: np.where(geom_ids == gid)[0].astype(np.int64)        # value = all dataset indices belonging to that geometry
            for gid in self.unique_gids}
        min_count = min(len(v) for v in self.idx_by_gid.values())     # Smallest number of samples among geometries.
        self.num_batches = max(1, min_count // self.samples_per_geom) # Number of batches we can draw before any geometry “runs out” of samples (roughly).

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return int(self.num_batches)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        pools = {}
        ptrs = {}
        for gid, idxs in self.idx_by_gid.items():
            pool = idxs.copy() 
            rng.shuffle(pool)   # For each geometry: shuffle its index list
            pools[gid] = pool   # store as pools
            ptrs[gid] = 0 # tracks where we are in that pool

        for _ in range(self.num_batches):
            batch = []
            for gid in self.unique_gids:
                # Get current pool and pointer for that geometry.
                pool = pools[gid]
                p = ptrs[gid]

                if p + self.samples_per_geom > len(pool): # if run past the end
                    pool = self.idx_by_gid[gid].copy()
                    rng.shuffle(pool) # reshuffle a fresh pool
                    pools[gid] = pool
                    p = 0 # restart pointer at 0

                batch.extend(pool[p : p + self.samples_per_geom].tolist()) 
                ptrs[gid] = p + self.samples_per_geom

            rng.shuffle(batch) # Shuffle combined batch (mix geometries)
            yield batch


def fit_normalisation_stats(dataset):
    """
    Compute normalisation statistics from an unnormalised dataset.
    """
    xyz = dataset.xyz
    sdf = dataset.sdf

    xyz_mean = xyz.mean(axis=0)
    xyz_std = xyz.std(axis=0)

    sdf_abs_max = float(np.abs(sdf).max())
    sdf_scale = sdf_abs_max if sdf_abs_max > 0 else 1.0

    return {
        "xyz_mean": xyz_mean,
        "xyz_std": xyz_std,
        "sdf_scale": np.array(sdf_scale, dtype=np.float32),
    }


def save_normalisation_stats(stats, path):
    """
    Save normalisation stats to disk as a .npz file.
    """
    path = Path(path)
    np.savez(path, xyz_mean=stats["xyz_mean"], xyz_std=stats["xyz_std"], sdf_scale=stats["sdf_scale"])
    print(f"[network1.data] Saved normalisation stats to {path}")


def load_normalisation_stats(path):
    """
    Load normalisation stats saved by save_normalisation_stats().
    """
    data = np.load(Path(path))
    return {
        "xyz_mean": data["xyz_mean"],
        "xyz_std": data["xyz_std"],
        "sdf_scale": data["sdf_scale"],
    }


def build_sdf_dataloader(cfg):
    """
    Build a DataLoader using Network1Config.

    Returns
    loader : DataLoader
        Yields (geom_id, xyz, sdf).
    num_geometries : int
        K inferred from the dataset.
    """
    raw_ds = ToolSDFDataset(sdf_dir=cfg.sdf_dir, transform=None)
    # Compute stats once and cache them, or load if already computed.
    stats_file = Path(cfg.stats_path)
    if not stats_file.is_file():
        stats = fit_normalisation_stats(raw_ds)
        save_normalisation_stats(stats, cfg.stats_path)
    else:
        stats = load_normalisation_stats(cfg.stats_path)

    transform = SDFTransform(
        xyz_mean=stats["xyz_mean"],
        xyz_std=stats["xyz_std"],
        sdf_scale=float(stats["sdf_scale"]),
    )

    ds = ToolSDFDataset(sdf_dir=cfg.sdf_dir, transform=transform) # create dataset with transform applied

    if cfg.use_balanced_batches:
        sampler = BalancedGeomBatchSampler(ds, samples_per_geom=cfg.samples_per_geom_in_batch, seed=cfg.seed)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
    else:
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    return loader, raw_ds.num_geometries


