"""
Offline preprocessing step for Network-1 (Auto-Decoder SDF).
- DDACS contains many simulations that may reuse the same tool geometry.
- Network-1 SDF training operates per-geometry, so we must:
    (a) group simulations into unique geometries
    (b) assign geometry IDs
    (c) choose one representative simulation per geometry

This script builds two parquet files:
1) meta.parquet
   - Full DDACS metadata with geometry parameters attached

2) geom_table.parquet
   - One row per unique tool geometry
   - Defines a stable geometry_id (0..K-1)
   - Stores a representative simulation ID (rep_ID) for each geometry,
     which is later used to extract meshes and generate SDF samples.
"""
import sys
from pathlib import Path

ROOT = Path("/home/RUS_CIP/st184634/implementation")  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddacs.pytorch import DDACSDataset
from atar_paper.Network1.config import MetadataConfig
import h5py
import numpy as np
from tqdm.auto import tqdm

cfg = MetadataConfig()
# Path to DDACS dataset root
DDACS_ROOT = cfg.ddacs_root
OUT_DIR = cfg.out_dir

def _read_geometry_parameters(h5_path):
    """
    Read the geometry parameter vector from a DDACS .h5 file.

    Parameters
    h5_path : Path
        Path to a DDACS simulation file.

    Returns
    geom : np.ndarray, shape (3,)
        Geometry_Parameters attribute stored in the file, typically:
        [geom_p1, geom_p2, geom_p3]
    """
    with h5py.File(h5_path, "r") as f:
        geom = f.attrs["Geometry_Parameters"]
    return np.asarray(geom)

def main():
    """
    Create meta.parquet and geom_table.parquet.

    Steps:
    1) Load the base DDACS metadata table
    2) For each simulation, read Geometry_Parameters from its .h5 file
    3) Attach these geometry parameters to the metadata table
    4) Group simulations by geometry-defining columns to form geom_table
    5) Assign a stable geometry_id (0..K-1)
    6) Save meta.parquet and geom_table.parquet
    """
    OUT_DIR.mkdir(exist_ok=True)

    # Load base DDACS metadata
    ds = DDACSDataset(DDACS_ROOT)
    meta = ds._metadata.copy()
    meta.index.name = "sim_id"

    print("[metadata_preparation] Metadata columns:", list(meta.columns))
    print("[metadata_preparation] First rows:")
    print(meta.head())

    # Attach Geometry_Parameters from each H5 file
    geom_list = []

    for sim_id in tqdm(meta.index,
                   desc="Reading geometry parameters",
                   total=len(meta),
                   mininterval=1.0,
                   smoothing=0.05,):
        # 'ID' is the simulation number used for the filename <ID>.h5
        sim_number = int(meta.loc[sim_id, "ID"])
        h5_path = DDACS_ROOT / "h5" / f"{sim_number}.h5"
        geom = _read_geometry_parameters(h5_path)
        geom_list.append(geom)

    geom_arr = np.vstack(geom_list)  # (num_sims, 3)
    meta[["geom_p1", "geom_p2", "geom_p3"]] = geom_arr

    print("\n[metadata_preparation] Metadata with geometry parameters (first rows):")
    print(meta.head())

    # Build compact geometry table: one row per unique geometry
    GROUP_COLS = cfg.geometry_group_cols

    geom_table = (
        meta.reset_index()
        .groupby(list(GROUP_COLS), sort=True)
        .agg(
            num_sims=("sim_id", "size"),
            rep_ID=("ID", "min"),  
        )
        .reset_index()
        .sort_values(list(GROUP_COLS))
        .reset_index(drop=True)
    )

    geom_table["geometry_id"] = np.arange(len(geom_table), dtype=np.int64)

    geom_table["split"] = "train"
    test_gid = int(geom_table["geometry_id"].max())  # "last one"
    geom_table.loc[geom_table["geometry_id"] == test_gid, "split"] = "test"

    print(f"\n[metadata_preparation] Using geometry_id={test_gid} as TEST, all others TRAIN.")

    print("\n[metadata_preparation] Split counts:")
    print(geom_table["split"].value_counts())

    print("\n[metadata_preparation] Split by topology (GEO_R,GEO_V,GEO_X):")
    print(geom_table.groupby(["GEO_R","GEO_V","GEO_X","split"]).size().unstack(fill_value=0))

    print("\n[metadata_preparation] Geometry table:")
    print(geom_table)

    # Save meta + geom_table
    meta_path = OUT_DIR / "meta.parquet"
    geom_path = OUT_DIR / "geom_table.parquet"
    meta.to_parquet(meta_path)
    geom_table.to_parquet(geom_path)

    print(f"\n[metadata_preparation] Saved:\n  - {meta_path}\n  - {geom_path}")


if __name__ == "__main__":
    main()
