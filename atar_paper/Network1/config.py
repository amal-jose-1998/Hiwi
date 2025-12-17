"""
Central configuration for:
- preprocessing (metadata + SDF generation)
- Network-1 (Auto-Decoder SDF) training
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass(frozen=True)
class SDFGenerationConfig:
    """
    Configuration for generating signed distance samples from DDACS tool meshes.
    """
    # Paths
    ddacs_root: Path = Path("/mnt/data/datasets/ddacs")
    meta_dir: Path = Path("/home/RUS_CIP/st184634/software_projects/atar_paper/prepared_metadata")
    out_dir: Path = Path("/home/RUS_CIP/st184634/software_projects/atar_paper/tool_sdf_samples")

    # Geometry extraction
    forming_timestep: int = 2
    tool_components: Tuple[str, ...] = ("die", "punch", "binder")

    # Surface perturbation (relative to tool size)
    surface_noise_scale: float = 0.02

    # Reproducibility
    seed: int = 0


@dataclass(frozen=True)
class MetadataConfig:
    """
    Configuration for geometry metadata preparation.
    """

    ddacs_root: Path = Path("/mnt/data/datasets/ddacs")
    out_dir: Path = Path("/home/RUS_CIP/st184634/software_projects/atar_paper/prepared_metadata")

    geometry_group_cols: Tuple[str, ...] = (
        "GEO_R",
        "GEO_V",
        "GEO_X",
        "geom_p1",
        "geom_p2",
        "geom_p3",
    )


@dataclass(frozen=True)
class Network1Config:
    """
    Central configuration for Network-1 (Auto-Decoder SDF) training.

    Network-1 learns an implicit Signed Distance Function (SDF) representation
    of tool geometries using an auto-decoder formulation (DeepSDF-style).

    Key ideas:
    - Geometry is represented implicitly via a learnable latent vector z_i
    - z_i is optimized jointly with the SDF decoder during pretraining
    - No explicit CAD parameters are used as inputs
    - Network-1 is later frozen and used inside a cascaded architecture
    """
    # Data
    sdf_dir: str = "/home/RUS_CIP/st184634/software_projects/atar_paper/tool_sdf_samples"
    stats_path: str = "/home/RUS_CIP/st184634/software_projects/atar_paper/network1_stats.npz"
    batch_size: int = 2048
    num_workers: int = 0
    shuffle: bool = True

    use_balanced_batches: bool = True
    samples_per_geom_in_batch: int = 256

    # Model
    latent_dim: int = 128
    hidden_dim: int = 256
    depth: int = 4

    # Optim
    lr: float = 1e-3
    weight_decay: float = 0.0
    latent_l2_weight: float = 1e-4

    # Training
    epochs: int = 50
    seed: int = 0
    device: str = "cuda"

    # Checkpoints
    ckpt_dir: str = "./checkpoints_network1"
    save_every: int = 5