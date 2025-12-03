from pathlib import Path
import numpy as np
import pandas as pd
import trimesh

from ddacs.utils import extract_mesh 

DDACS_ROOT = Path("/mnt/data/datasets/ddacs") # contains all .h5 simulation files
META_DIR = Path("./prepared_metadata") # contains the parquet files generated earlier
SDF_OUT_DIR = Path("./tool_sdf_samples") # location where the SDF samples will be stored

FINAL_FORMING_TIMESTEP = 2   # timestep where tools and blank exist (pre-springback)
TOOL_COMPONENTS = ["die", "punch", "binder"]  # everything except blank


def build_tool_mesh(h5_path, timestep=FINAL_FORMING_TIMESTEP):
    """
    Combine die + punch + binder meshes into a single tool mesh.
    Returns:
        vertices: (N_total, 3)
        triangles: (F_total, 3) 
    """
    all_vertices = []
    all_faces = []
    offset = 0

    for comp in TOOL_COMPONENTS:
        v, f = extract_mesh(h5_path, comp, timestep=timestep)
        all_vertices.append(v)
        all_faces.append(f + offset)
        offset += v.shape[0]

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)
    return vertices, faces


def sample_sdf_for_mesh(vertices, faces, num_samples=200000, surface_bias=0.7):
    """
    Sample 3D points around the mesh and compute signed distances using trimesh.
    Returns:
        points: (N, 3)
        sdf:    (N,)
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False) # Makes geometry ready for distance queries.

    # Bounding box + margin
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = (maxs - mins) / 2.0
    margin = 0.2 * np.max(extent) # Bounding box expanded by 20% to give space around the tool.
    # 3D volume where points will be randomly sampled.
    mins_box = center - extent - margin
    maxs_box = center + extent + margin

    # How many samples near surface vs uniform
    n_surface = int(num_samples * surface_bias) # 70% near surface => surface_bias=0.7
    n_uniform = num_samples - n_surface

    # Near surface: sample mesh surface and perturb
    surf_points, _ = trimesh.sample.sample_surface(mesh, n_surface)
    noise = np.random.normal(scale=0.02 * np.max(extent), size=surf_points.shape) # Adds small jitter so we have points just inside/outside the surface.
    pts_surface = surf_points + noise

    # Uniform in bounding box
    pts_uniform = np.random.uniform(mins_box, maxs_box, size=(n_uniform, 3))

    points = np.vstack([pts_surface, pts_uniform])

    # Distances to surface
    _, dist, _ = mesh.nearest.on_surface(points)  # unsigned distance

    # Inside/outside: trimesh.contains
    inside = mesh.contains(points)  # boolean (N,)
    sign = np.where(inside, -1.0, 1.0) # Points inside the mesh => negative SDF; Points outside => positive SDF

    sdf = dist * sign
    return points.astype(np.float32), sdf.astype(np.float32)


def main():
    SDF_OUT_DIR.mkdir(exist_ok=True)

    geom_table = pd.read_parquet(META_DIR / "geom_table.parquet")

    print("Generating SDF samples for each geometry variant...\n")

    for _, row in geom_table.iterrows():
        geom_id = int(row["geometry_id"])
        rep_id = int(row["rep_ID"])  # representative simulation ID

        h5_path = DDACS_ROOT / "h5" / f"{rep_id}.h5"
        print(f"Geometry {geom_id}: using rep_ID={rep_id}, h5={h5_path}")

        vertices, faces = build_tool_mesh(h5_path)

        print(f"  Mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

        points, sdf = sample_sdf_for_mesh(vertices, faces, num_samples=200_000)

        out_path = SDF_OUT_DIR / f"tool_geom{geom_id}_sdf.npz" # training data for Network 1
        np.savez(out_path, points=points, sdf=sdf, geom_id=geom_id)
        print(f"  Saved SDF samples to {out_path} (N={points.shape[0]})\n")

    print("Done.")


if __name__ == "__main__":
    main()
