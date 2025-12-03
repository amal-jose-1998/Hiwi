from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class ToolSDFDataset(Dataset):
    """
    Dataset for Network 1 (conditional SDF).

    Each sample:
        x: np.ndarray, shape (9,)
           [GEO_R, GEO_V, GEO_X, geom_p1, geom_p2, geom_p3, x, y, z]
        y: np.ndarray, shape (1,)
           [sdf]
    """

    def __init__(self, sdf_dir, geom_table, transform=None):
        self.sdf_dir = Path(sdf_dir)
        self.sdf_files = sorted(self.sdf_dir.glob("tool_geom*_sdf.npz")) # Load all 6 geometry SDF files.
        if not self.sdf_files:
            raise RuntimeError(f"No SDF npz files found in {self.sdf_dir}")

        # index by geometry_id
        self.geom_table = geom_table.reset_index(drop=True).set_index("geometry_id")
        self.transform = transform

        inputs = []
        labels = []

        for path in self.sdf_files:
            data = np.load(path)
            points = data["points"]  # (N,3)
            sdf = data["sdf"]        # (N,)
            geom_id = int(data["geom_id"])

            info = self.geom_table.loc[geom_id]
            # get the 6-dimensional condition vector.
            GEO_R, GEO_V, GEO_X = info["GEO_R"], info["GEO_V"], info["GEO_X"]
            p1, p2, p3 = info["geom_p1"], info["geom_p2"], info["geom_p3"]

            geom_vec = np.array(
                [GEO_R, GEO_V, GEO_X, p1, p2, p3],
                dtype=np.float32,
            )  # (6,)

            N = points.shape[0]
            geom_tile = np.repeat(geom_vec[None, :], N, axis=0)  # (N,6)
            xyz = points.astype(np.float32)                      # (N,3)
            # input/output pairs the network needs.
            x_vec = np.concatenate([geom_tile, xyz], axis=1)     # (N,9) Concatenate geometry + xyz
            y_vec = sdf.astype(np.float32)[:, None]              # (N,1)

            inputs.append(x_vec)
            labels.append(y_vec)

        self.inputs = np.vstack(inputs)   # (M,9)
        self.labels = np.vstack(labels)   # (M,1)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y


class SDFTransform:
    """
    Normalisation transform for ToolSDFDataset.

    - Leaves [GEO_R, GEO_V, GEO_X] as is (0/1)
    - Normalises [geom_p1, geom_p2, geom_p3] using mean/std
    - Normalises [x, y, z] using mean/std
    - Scales sdf by dividing by sdf_scale
    """

    def __init__(self, geom_mean, geom_std, xyz_mean, xyz_std, sdf_scale):
        self.geom_mean = geom_mean
        self.geom_std = np.where(geom_std == 0, 1.0, geom_std)
        self.xyz_mean = xyz_mean
        self.xyz_std = np.where(xyz_std == 0, 1.0, xyz_std)
        self.sdf_scale = sdf_scale if sdf_scale != 0 else 1.0

    def __call__(self, x, y):
        # x: [onehot(3), geom(3), xyz(3)]
        onehot = x[:3]
        geom = x[3:6]
        xyz = x[6:9]

        geom_norm = (geom - self.geom_mean) / self.geom_std
        xyz_norm = (xyz - self.xyz_mean) / self.xyz_std

        x_new = np.concatenate([onehot, geom_norm, xyz_norm], axis=0).astype(np.float32)
        y_new = (y / self.sdf_scale).astype(np.float32)

        return x_new, y_new


def fit_normalisation_stats(dataset):
    """
    Compute normalisation statistics from an unnormalised ToolSDFDataset.
    """
    X = dataset.inputs  # (M,9)
    Y = dataset.labels  # (M,1)

    geom = X[:, 3:6]  # geom_p1..3
    xyz = X[:, 6:9]   # xyz

    # mean/std from the raw dataset
    geom_mean = geom.mean(axis=0)
    geom_std = geom.std(axis=0)
    xyz_mean = xyz.mean(axis=0)
    xyz_std = xyz.std(axis=0)

    sdf_abs_max = np.abs(Y).max()
    sdf_scale = float(sdf_abs_max) if sdf_abs_max > 0 else 1.0 # sdf_scale = max(|sdf|)

    return {
        "geom_mean": geom_mean,
        "geom_std": geom_std,
        "xyz_mean": xyz_mean,
        "xyz_std": xyz_std,
        "sdf_scale": sdf_scale,
    }

"""
When one resume training or evaluate on test data, one must use the same normalisation.
These functions let one save/load them from a .npz file.
"""
def save_normalisation_stats(stats, path):
    path = Path(path)
    np.savez(
        path,
        geom_mean=stats["geom_mean"],
        geom_std=stats["geom_std"],
        xyz_mean=stats["xyz_mean"],
        xyz_std=stats["xyz_std"],
        sdf_scale=stats["sdf_scale"],
    )
    print(f"Saved normalisation stats to {path}")


def load_normalisation_stats(path):
    path = Path(path)
    data = np.load(path)
    return {
        "geom_mean": data["geom_mean"],
        "geom_std": data["geom_std"],
        "xyz_mean": data["xyz_mean"],
        "xyz_std": data["xyz_std"],
        "sdf_scale": float(data["sdf_scale"]),
    }


def build_sdf_dataloader(sdf_dir, geom_table_path, stats_path, batch_size=65536, shuffle=True, num_workers=0):
    """
    High-level helper for training:

      - loads geom_table
      - builds raw ToolSDFDataset
      - fits or loads normalisation stats
      - builds normalised ToolSDFDataset
      - returns DataLoader
    """
    geom_table = pd.read_parquet(geom_table_path)

    # raw dataset, no transform
    raw_ds = ToolSDFDataset(sdf_dir=sdf_dir, geom_table=geom_table, transform=None)

    stats_path = Path(stats_path)
    if not stats_path.is_file():
        stats = fit_normalisation_stats(raw_ds)
        save_normalisation_stats(stats, stats_path)
    else:
        stats = load_normalisation_stats(stats_path)

    transform = SDFTransform(
        geom_mean=stats["geom_mean"],
        geom_std=stats["geom_std"],
        xyz_mean=stats["xyz_mean"],
        xyz_std=stats["xyz_std"],
        sdf_scale=stats["sdf_scale"],
    )

    ds = ToolSDFDataset(sdf_dir=sdf_dir, geom_table=geom_table, transform=transform)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
