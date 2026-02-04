"""Dataset and dataloader utilities for Network-1 (Auto-Decoder SDF)."""

from pathlib import Path
from typing import List
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

stats = {
    "xyz_mean": np.zeros(3, dtype=np.float32),
    "xyz_std": np.ones(3, dtype=np.float32),
    "sdf_scale": np.array(1.0, dtype=np.float32),
}


class ToolSDFDataset(Dataset):
    """
    Dataset for Network-1 (Auto-Decoder SDF), component-aware.

    Each sample:
        geom_id : torch.long scalar in [0, K-1]
            Geometry index (used to look up latent vector in nn.Embedding).
        comp_id : torch.long scalar in [0, C-1]
            Component index (die/punch/binder).
        xyz : torch.float32 tensor, shape (3,)
            Query point.
        sdf : torch.float32 tensor, shape (1,)
            Ground-truth signed distance.
    """

    def __init__(self, sdf_dir, transform=None, components=None, allowed_geom_ids=None, gid_remap=None):
        self.sdf_dir = Path(sdf_dir)
        self.sdf_files = sorted(self.sdf_dir.glob("tool_geom*_*_sdf.npz"))
        self.components = list(components) if components is not None else ["die", "punch", "binder"]
        # comp ids are:
        # die → 0
        # punch → 1
        # binder → 2
        self.comp_to_id = {c: i for i, c in enumerate(self.components)}

        if not self.sdf_files:
            raise RuntimeError(f"No SDF npz files found in {self.sdf_dir} (expected tool_geom*_*_sdf.npz)")

        self.transform = transform
        self.gid_remap = gid_remap

        # Load and stack all samples
        xyz_all = []
        sdf_all = []
        geom_id_all = []
        comp_id_all = []

        allowed = None if allowed_geom_ids is None else set(map(int, allowed_geom_ids)) # train dataset loads only train geometry IDs, test dataset loads only test geometry IDs.

        for path in self.sdf_files:
            data = np.load(path, allow_pickle=True)

            points = data["points"].astype(np.float32)          # (N,3)
            sdf = data["sdf"].astype(np.float32)[:, None]       # (N,1)
            geom_id = int(data["geom_id"])

            if allowed is not None and geom_id not in allowed:
                continue

            if "comp" in data:
                comp = str(data["comp"])
            else:
                # tool_geom{gid}_{comp}_sdf.npz
                comp = path.stem.split("_")[-2]

            if comp not in self.comp_to_id:
                raise ValueError(
                    f"Found comp='{comp}' in {path.name}, but components={self.components}. "
                    "Make sure cfg.tool_components matches preprocessing."
                )
            comp_id = int(self.comp_to_id[comp])

            if self.gid_remap is not None:
                geom_id = int(self.gid_remap[geom_id])

            # latent embedding has size num_train_geoms * num_components.
            N = points.shape[0]
            geom_ids = np.full((N,), geom_id, dtype=np.int64)
            comp_ids = np.full((N,), comp_id, dtype=np.int64)

            xyz_all.append(points)
            sdf_all.append(sdf)
            geom_id_all.append(geom_ids)
            comp_id_all.append(comp_ids)

        if not xyz_all:
            raise RuntimeError("No SDF samples matched allowed_geom_ids.")

        # M = total number of SDF points across all (geometry, component) files
        self.xyz = np.vstack(xyz_all)                    # (M,3)
        self.sdf = np.vstack(sdf_all)                    # (M,1)
        self.geom_id = np.concatenate(geom_id_all)       # (M,)
        self.comp_id = np.concatenate(comp_id_all)       # (M,)

        self.num_geometries = int(self.geom_id.max()) + 1
        self.num_components = len(self.components)

        self.pair_id = self.geom_id.astype(np.int64) * self.num_components + self.comp_id.astype(np.int64) # every point knows which (geometry, component) pair it belongs to.

    def __len__(self):
        return int(self.xyz.shape[0])

    def __getitem__(self, idx):
        geom_id = int(self.geom_id[idx])
        comp_id = int(self.comp_id[idx])
        xyz = self.xyz[idx]
        sdf = self.sdf[idx]

        if self.transform:
            xyz, sdf = self.transform(xyz, sdf)

        return (
            torch.tensor(geom_id, dtype=torch.long),
            torch.tensor(comp_id, dtype=torch.long),
            torch.from_numpy(xyz).float(),
            torch.from_numpy(sdf).float(),
        )


class BalancedGeomBatchSampler(Sampler[List[int]]):
    """
    Balanced batch sampler over (geom_id, comp_id) PAIRS.

    Effective batch size:
        batch_size = pairs_per_batch * samples_per_pair

    Requirements:
        dataset.pair_id : np.ndarray of shape (M,)
            Pair id per sample (geom_id * C + comp_id).
    """

    def __init__(self, dataset, geoms_per_batch, samples_per_geom, seed=0):
        # - geoms_per_batch == pairs_per_batch
        # - samples_per_geom == samples_per_pair
        self.dataset = dataset
        self.pairs_per_batch = int(geoms_per_batch)
        self.samples_per_pair = int(samples_per_geom)
        self.seed = int(seed)
        self.epoch = 0

        if not hasattr(dataset, "pair_id"):
            raise RuntimeError("Dataset must expose dataset.pair_id for BalancedGeomBatchSampler (component-aware).")

        # 1. collect all unique pair_ids
        pair_ids = dataset.pair_id 
        self.unique_pids = np.unique(pair_ids).astype(np.int64).tolist()
        if len(self.unique_pids) == 0:
            raise RuntimeError("No pairs found in dataset.pair_id")

        # Clamp pairs_per_batch to available pairs
        self.pairs_per_batch = max(1, min(self.pairs_per_batch, len(self.unique_pids)))

        # 2. For each (geometry, component) pair (pid), store the dataset indices of all points that belong to that pair. 
        self.idx_by_pid = {
            pid: np.where(pair_ids == pid)[0].astype(np.int64)
            for pid in self.unique_pids
        }

        self.num_batches = int(np.ceil(len(self.unique_pids) / self.pairs_per_batch))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return int(self.num_batches)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        # 3. Shuffle pair order in each epoch
        pids = np.array(self.unique_pids, dtype=np.int64)
        rng.shuffle(pids)

        # Create shuffled pools per pair and pointers
        pools = {}
        ptrs = {}
        for pid in pids:
            # 4. for each pid, shuffle its point indices (“pool”)
            pid = int(pid)
            pool = self.idx_by_pid[pid].copy()
            rng.shuffle(pool)
            pools[pid] = pool
            ptrs[pid] = 0

        # Split pids into batches
        for b in range(self.num_batches):
            start = b * self.pairs_per_batch
            end = min((b + 1) * self.pairs_per_batch, len(pids))
            batch_pids = pids[start:end]

            batch = []
            # 5. yield batches by taking samples_per_pair from each pid in the batch
            for pid in batch_pids:
                pid = int(pid)
                pool = pools[pid]
                p = ptrs[pid]

                # 6. if a pool runs out, reshuffle and restart it
                if p + self.samples_per_pair > len(pool):
                    pool = self.idx_by_pid[pid].copy()
                    rng.shuffle(pool)
                    pools[pid] = pool
                    p = 0

                batch.extend(pool[p : p + self.samples_per_pair].tolist())
                ptrs[pid] = p + self.samples_per_pair

            rng.shuffle(batch)
            yield batch


def split_indices_by_geometry(geom_id, val_frac, seed):
    """
    Stratified (by group id) point-level split.

    Returns train_idx, val_idx arrays of indices into the dataset arrays.
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0,1)")

    rng = np.random.default_rng(int(seed))
    train_parts = []
    val_parts = []

    for gid in np.unique(geom_id): # For each unique pair_id
        idx = np.where(geom_id == gid)[0]
        rng.shuffle(idx) # shuffle its indices

        n = len(idx)
        n_val = int(np.round(val_frac * n)) # put ~val_frac into val, rest into train
        # keep at least 1 point in each split (if possible)
        n_val = max(1, min(n_val, n - 1))

        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])

    train_idx = np.concatenate(train_parts).astype(np.int64)
    val_idx = np.concatenate(val_parts).astype(np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx # Val will contain points from every pair the model sees in training.


def subset_toolsdf_dataset(ds, idx):
    """
    Create a lightweight subset of ToolSDFDataset that preserves ds.geom_id, ds.comp_id, ds.xyz, ds.sdf arrays
    """
    idx = np.asarray(idx, dtype=np.int64)
    out = copy.copy(ds)
    out.xyz = ds.xyz[idx]
    out.sdf = ds.sdf[idx]
    out.geom_id = ds.geom_id[idx]
    out.comp_id = ds.comp_id[idx]
    out.pair_id = ds.pair_id[idx]
    out.num_geometries = int(out.geom_id.max()) + 1  
    out.num_components = ds.num_components
    return out


def build_sdf_dataloader(cfg):
    """
    Build a DataLoader using Network1Config.

    Returns:
      train_loader, val_loader, test_loader, n_train_geoms, n_test_geoms, stats

    Notes:
      - train/val are point-level splits within TRAIN geometries (for early stopping).
      - test_loader (held-out geometries) is NOT meant for forward-pass validation.
        Use latent-optimisation evaluation script for test.
    """
    geom_table = pd.read_parquet(cfg.geom_table_path)
    # splitting is done at geometry level
    train_gids = geom_table.loc[geom_table["split"] == "train", "geometry_id"].to_list()
    test_gids = geom_table.loc[geom_table["split"] == "test", "geometry_id"].to_list()

    train_gid_to_local = {int(gid): i for i, gid in enumerate(train_gids)} # embedding table is sized for train geometries only.

    # full train dataset
    train_full = ToolSDFDataset(
        cfg.sdf_dir,
        transform=None,
        components=getattr(cfg, "tool_components", ["die", "punch", "binder"]),
        allowed_geom_ids=train_gids,
        gid_remap=train_gid_to_local,
    )

    # Point-level train/val split inside training geometries
    val_frac = float(getattr(cfg, "val_frac", 0.1))
    train_idx, val_idx = split_indices_by_geometry(train_full.pair_id, val_frac=val_frac, seed=int(cfg.seed))
    train_ds = subset_toolsdf_dataset(train_full, train_idx)
    val_ds = subset_toolsdf_dataset(train_full, val_idx)

    # Build test dataset (test geometries only)
    # no gid_remap here because we don’t use test_loader for direct forward validation. Ther true test evaluation is latent-optimization (evaluate.py), 
    # where we optimize a fresh z.
    test_ds = ToolSDFDataset(
        cfg.sdf_dir,
        transform=None,
        components=getattr(cfg, "tool_components", ["die", "punch", "binder"]),
        allowed_geom_ids=test_gids,
    )

    # Create DataLoaders
    if cfg.use_balanced_batches:
        sampler = BalancedGeomBatchSampler(
            train_ds,
            geoms_per_batch=int(getattr(cfg, "geoms_per_batch", len(np.unique(train_ds.pair_id)))),
            samples_per_geom=int(cfg.samples_per_geom_in_batch),
            seed=int(cfg.seed),
        )
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, len(train_gids), len(test_gids), stats
