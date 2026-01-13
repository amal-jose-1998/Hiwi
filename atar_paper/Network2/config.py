from pathlib import Path
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class MeshFromLatentConfig:
    """
    Configuration for extracting a surface mesh from a latent code using Network-1.

    Attributes
    grid_res:
        Resolution N of the cubic evaluation grid (N x N x N).
        Higher values produce smoother meshes but increase compute/memory.
    grid_min, grid_max:
        World-coordinate bounds of the evaluation cube (inclusive endpoints).
    iso_level:
        Iso-value of the SDF surface to extract. For SDFs, the surface is SDF=0.0.
    chunk_size:
        Number of query points evaluated per forward pass chunk to avoid GPU OOM.
        Total points is N^3.
    device:
        Device string used for decoder evaluation ("cuda" or "cpu").
    dtype:
        Floating point dtype used during decoder evaluation (typically float32).
    
    """
    grid_res: int = 128                       
    grid_min: float = -0.6                    
    grid_max: float = 0.6                    
    iso_level: float = 0.0                    
    chunk_size: int = 262144                 # ≈ 1/8 of the grid; points per forward chunk (avoid OOM)
    device: str = "cuda"                      
    dtype: torch.dtype = torch.float32        

@dataclass(frozen=True)
class DepthProjectionConfig:
    """
    Configuration for mesh -> depth map projection used as Network-2 input.

    Attributes
    resolution :
        int. Output depth image size (H=W=resolution).
    n_surface_points :
        int. Number of surface points sampled from the mesh for rasterization.
        Increase if depth maps look sparse.
    depth_mode :
        str. Aggregation mode per pixel: "min" or "max".
        - "max" corresponds to the farthest surface along +axis
        - "min" corresponds to the nearest surface along +axis
    fill_value :
        float. Value used for pixels with no samples.
    margin :
        float. Extra padding when auto-computing plane bounds from sampled points.
    plane_min_yz, plane_max_yz :
        Optional np.ndarray of shape (2,) for fixed YZ bounds. If None, auto bounds.
    plane_min_xy, plane_max_xy :
        Optional np.ndarray of shape (2,) for fixed XY bounds. If None, auto bounds.
    """
    resolution: int = 256
    n_surface_points: int = 200000
    depth_mode: str = "max"
    fill_value: float = 0.0
    margin: float = 0.02
    plane_min_yz = None
    plane_max_yz = None
    plane_min_xy = None
    plane_max_xy = None


@dataclass(frozen=True)
class BuildSurrogateInputsConfig:
    """
    Configuration for building cached surrogate inputs for Network-2.

    Paths
    network1_ckpt :
        Path to trained Network-1 checkpoint (.ckpt/.pt).
    geom_table_path :
        Path to geom_table.parquet (one row per geometry_id).
    meta_path :
        Path to meta.parquet (one row per simulation ID).
    sdf_dir :
        Directory containing tool_geom{gid}_sdf.npz (for latent inference fallback).
    out_dir :
        Output directory for *.npz and manifest.

    Latent handling
    use_latent_table_if_available :
        If checkpoint contains latent embedding table, use it when possible.
    infer_latent_if_missing :
        If geometry_id not covered by latent table, fit z by optimizing against SDF samples.

    Geometry vs process parameter separation
    geom_param_cols :
        Geometry parameters are taken ONLY from geom_table row (per geometry_id).
    proc_param_cols :
        Process parameters are taken ONLY from meta row at rep_ID (representative simulation).
    proc_exclude_cols :
        Safety exclusion list so geometry-identifying cols never leak into proc_params.
    """
    # -------------------------------
    # Paths (set these to your repo)
    # -------------------------------
    network1_ckpt: Path = Path("./checkpoints/network1/last.ckpt")
    geom_table_path: Path = Path("./prepared_metadata/geom_table.parquet")
    meta_path: Path = Path("./prepared_metadata/meta.parquet")
    sdf_dir: Path = Path("./tool_sdf_samples")
    out_dir: Path = Path("./network2_surrogate_inputs")

    # -------------------------------
    # Reproducibility
    # -------------------------------
    seed: int = 0

    # -------------------------------
    # Latent retrieval / inference
    # -------------------------------
    use_latent_table_if_available: bool = True
    infer_latent_if_missing: bool = True

    latent_steps: int = 800
    latent_lr: float = 1e-2
    latent_init_std: float = 0.012
    fit_frac: float = 0.5
    use_clamped_loss: bool = True

    # -------------------------------
    # Column policy (IMPORTANT)
    # -------------------------------
    # Comes from geom_table (per geometry_id), not really present in the papaer
    geom_param_cols = ("GEO_R", "GEO_V", "GEO_X", "geom_p1", "geom_p2", "geom_p3")

    # Comes from meta at rep_ID (per representative simulation)
    proc_param_cols = ("RAD", "MAT", "FC", "SHTK", "BF")

    # Never allow these inside proc_params
    proc_exclude_cols = (
        "ID", "rep_ID", "geometry_id", "split",
        "GEO_R", "GEO_V", "GEO_X",
        "geom_p1", "geom_p2", "geom_p3",
    )