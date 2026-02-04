"""
Offline preprocessing script for Network-1 (Auto-Decoder SDF).

This script:
1. Extracts tool meshes (die + punch + binder) from DDACS simulation files
2. Combines them into a single tool mesh per geometry
3. Samples 3D query points around the tool
4. Computes signed distance values (SDF)
5. Saves the result to disk as .npz files

The generated files are used as ground-truth training data
for Network-1 (explicit SDF regression / DeepSDF-like).
"""
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
import sys
from pathlib import Path

ROOT = Path("/home/RUS_CIP/st184634/software_projects")  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddacs.utils import extract_mesh 

from atar_paper.Network1.config import SDFGenerationConfig
cfg = SDFGenerationConfig()

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DDACS_ROOT = cfg.ddacs_root
META_DIR = cfg.meta_dir
SDF_OUT_DIR = cfg.out_dir

FINAL_FORMING_TIMESTEP = cfg.forming_timestep   
TOOL_COMPONENTS = cfg.tool_components  

# =============================================================================
# Mesh construction
# =============================================================================
def build_component_mesh(h5_path, comp, timestep=FINAL_FORMING_TIMESTEP):
    """
    Extract ONE tool component mesh (die/punch/binder) from a DDACS .h5 file,
    then center+scale so its XY edge length = 1 (component-wise normalisation).
    """
    v, f = extract_mesh(h5_path, comp, timestep=timestep)

    # Center first
    center = (v.min(axis=0) + v.max(axis=0)) / 2.0
    v_centered = v - center

    # XY bbox edge length proxy
    xy = v_centered[:, :2]
    tool_edge = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    if tool_edge <= 0:
        raise ValueError(f"Invalid tool_edge for comp={comp} in {h5_path}")

    v_scaled = v_centered / tool_edge

    return v_scaled, f, {
        "tool_edge": tool_edge,
        "center": center.astype(np.float32),
    }

# =============================================================================
# SDF sampling
# =============================================================================
def sample_sdf_for_mesh(vertices, faces, rng=None):
    """
    Sample 3D query points around a mesh and compute signed distances.
      - 3000 surface points (sdf = 0)
      - 3000 low-noise perturbed surface points
      - 3000 high-noise perturbed surface points
      - 3000 uniform points in unit cube 

    SDF definition:
    - Distance = Euclidean distance to nearest surface point
    - Sign = sign( n_s · (x - x_s) ), where:
        * x_s : nearest surface point
        * n_s : surface normal at x_s

    Parameters
    vertices : np.ndarray, shape (N, 3)
        Mesh vertices.
    faces : np.ndarray, shape (F, 3)
        Mesh faces.
    
    Returns
    points : (12000, 3) np.ndarray
        Sampled query points.
    sdf : (12000,) np.ndarray
        Corresponding signed distance values.
    """
    rng = np.random.default_rng() if rng is None else rng

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False) # Makes geometry ready for distance queries.

    n_surface = 3000
    n_low = 3000
    n_high = 3000
    n_uniform = 3000

    sigma_low = cfg.surface_noise_scale * 0.25
    sigma_high = cfg.surface_noise_scale

    # Surface points + corresponding face normals
    surf_pts, face_idx = trimesh.sample.sample_surface(mesh, n_surface)
    surf_normals = mesh.face_normals[face_idx]
    surf_normals /= np.linalg.norm(surf_normals, axis=1, keepdims=True) + 1e-12

    # Perturbed off-surface points: low/high noise
    idx_low = rng.choice(n_surface, size=n_low, replace=True)
    pts_low = surf_pts[idx_low] + rng.normal(scale=sigma_low, size=(n_low, 3))
    idx_high = rng.choice(n_surface, size=n_high, replace=True)
    pts_high = surf_pts[idx_high] + rng.normal(scale=sigma_high, size=(n_high, 3))

    # Uniform unit cube (edge length = 1), centered at origin
    pts_uniform = rng.uniform(-0.5, 0.5, size=(n_uniform, 3))

    off_pts = np.vstack([pts_low, pts_high, pts_uniform])

    # Nearest surface point via KD-tree
    tree = cKDTree(surf_pts)
    dists, idx = tree.query(off_pts, k=1, workers=-1)

    nearest_s = surf_pts[idx]
    nearest_n = surf_normals[idx]

    v = off_pts - nearest_s
    sign = np.sign(np.einsum("ij,ij->i", nearest_n, v))
    sign[sign == 0.0] = 1.0

    sdf_off = dists * sign

    # Combine: surface points have sdf=0
    points = np.vstack([surf_pts, off_pts]).astype(np.float32)
    sdf = np.concatenate([
        np.zeros(n_surface, dtype=np.float32),
        sdf_off.astype(np.float32)
    ])

    return points, sdf

# =============================================================================
# Main
# =============================================================================
def main():
    """
    Generate SDF samples for all unique tool geometries.

    For each geometry defined in geom_table.parquet:
    - Select a representative simulation
    - Extract tool mesh
    - Sample SDF points
    - Save results to tool_geom{geom_id}_sdf.npz
    """
    SDF_OUT_DIR.mkdir(exist_ok=True)

    geom_table = pd.read_parquet(META_DIR / "geom_table.parquet")

    print("Generating SDF samples for each geometry variant...\n")

    for _, row in tqdm(geom_table.iterrows(), total=len(geom_table), desc="Generating SDF per geometry/component"):
        geom_id = int(row["geometry_id"])
        rep_id = int(row["rep_ID"])  # representative simulation ID

        h5_path = DDACS_ROOT / "h5" / f"{rep_id}.h5"
        print(f"Geometry {geom_id}: using rep_ID={rep_id}, h5={h5_path}")

        for comp in TOOL_COMPONENTS:
            vertices, faces, meta = build_component_mesh(h5_path, comp)

            print(f"  [{comp}] Mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

            rng = np.random.default_rng(cfg.seed + geom_id * 100 + (abs(hash(comp)) % 97))

            points, sdf = sample_sdf_for_mesh(vertices, faces, rng=rng)

            out_path = SDF_OUT_DIR / f"tool_geom{geom_id}_{comp}_sdf.npz"
            np.savez(
                out_path,
                points=points,
                sdf=sdf,
                geom_id=geom_id,
                comp=comp,
                tool_edge=meta["tool_edge"],
                center=meta["center"],
            )
            print(f"  [{comp}] Saved SDF samples to {out_path} (N={points.shape[0]})\n")

    print("Done.")


if __name__ == "__main__":
    main()
