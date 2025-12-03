from pathlib import Path
from ddacs.pytorch import DDACSDataset
import h5py
import numpy as np
import pandas as pd

# Path to DDACS dataset root
DDACS_ROOT = Path("/mnt/data/datasets/ddacs")
OUT_DIR = Path("./prepared_metadata")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Load base DDACS metadata
    ds = DDACSDataset(DDACS_ROOT)
    meta = ds._metadata.copy()
    meta.index.name = "sim_id"

    print("Metadata columns:", meta.columns)
    print(meta.head())

    # Attach Geometry_Parameters from each H5 file
    geom_list = []

    for sim_id in meta.index:
        sim_number = int(meta.loc[sim_id, "ID"])
        h5_path = DDACS_ROOT / "h5" / f"{sim_number}.h5"
        with h5py.File(h5_path, "r") as f:
            geom = f.attrs["Geometry_Parameters"]  # [geom_p1, geom_p2, geom_p3]
        geom_list.append(geom)

    geom_arr = np.vstack(geom_list)
    meta[["geom_p1", "geom_p2", "geom_p3"]] = geom_arr

    print("\nMetadata with geometry info (first 5):")
    print(meta.head())

    # Build compact geometry table: one row per unique geometry
    group_cols = ["GEO_R", "GEO_V", "GEO_X", "geom_p1", "geom_p2", "geom_p3"]

    geom_table = (
        meta.reset_index()
        .groupby(group_cols)
        .agg(
            num_sims=("sim_id", "size"),
            rep_ID=("ID", "first"),  # representative simulation ID for this geometry, to get the tool mesh
        )
        .reset_index()
    )

    # Assign stable geometry_id 0..N-1
    geom_table["geometry_id"] = range(len(geom_table))

    print("\nGeometry table:")
    print(geom_table)

    # Save meta + geom_table
    meta.to_parquet(OUT_DIR / "meta.parquet")
    geom_table.to_parquet(OUT_DIR / "geom_table.parquet")

    print(f"\nSaved meta and geom_table to {OUT_DIR}")


if __name__ == "__main__":
    main()
