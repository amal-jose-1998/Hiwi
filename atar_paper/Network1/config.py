"""
Central configuration for:
- preprocessing (metadata + SDF generation)
- Network-1 (Auto-Decoder SDF) training
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Literal

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

    seed: int = 0
    test_frac: float = 0.2


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
    geom_table_path: str = "/home/RUS_CIP/st184634/software_projects/atar_paper/prepared_metadata/geom_table.parquet"

    # Sampling/batching
    batch_size: int = 2048
    num_workers: int = 8
    shuffle: bool = True
    
    use_balanced_batches: bool = True
    geoms_per_batch: int = 5
    samples_per_geom_in_batch: int = 256

    # Model
    latent_dim: int = 128
    hidden_dim: int = 256
    depth: int = 4
    latent_init_std: float = 0.012
    activation: Literal["relu", "softplus"] = "relu"
    truncation_delta: float = 0.05
    use_skip: bool = True
    skip_layer: int | None = 1

    # Optim
    lr: float = 1e-4
    weight_decay: float = 0.0
    latent_l2_weight: float = 1e-4

    # LR schedule 
    lr_step_size_epochs: int = 200
    lr_gamma: float = 0.5
    lr_min: float = 5e-6

    # Training
    epochs: int = 2000
    seed: int = 0
    device: str = "cuda"

    # Checkpoints
    ckpt_dir: str = "atar_paper/checkpoints_network1"
    save_every: int = 5

    val_frac: float = 0.1
    
    # ----------------------------
    # Evaluation (Network-1)
    # ----------------------------
    eval_ckpt_path: str = "atar_paper/checkpoints_network1/best.ckpt"
    eval_fit_frac: float = 0.5

    # latent optimization (per held-out geometry)
    eval_latent_steps: int = 800
    eval_latent_lr: float = 1e-2
    eval_latent_init_std: float = 0.012
    eval_seed: int = 0

    # reporting
    eval_out_csv: str = "atar_paper/network1_test_latent_opt.csv"

    # loss used for fitting z and reporting
    # "clamped" matches training-style truncation; "l1" is unclamped abs error
    eval_loss_mode: Literal["clamped", "l1"] = "clamped"
    
    
    
