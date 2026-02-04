import torch
import numpy as np
from pathlib import Path
import pandas as pd
import trimesh
from skimage import measure

import sys
from pathlib import Path

ROOT = Path("/home/RUS_CIP/st184634/software_projects")  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atar_paper.Network1.config import Network1Config
from atar_paper.Network1.model import Network1SDF


# ------------------------------------------------------------
# Load full Network-1 model (decoder + latents) from checkpoint
# ------------------------------------------------------------
def load_trained_model(ckpt_path, cfg, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]

    # number of TRAIN geometries (same as during training)
    geom_table = pd.read_parquet(cfg.geom_table_path)
    n_train_geoms = int((geom_table["split"] == "train").sum())

    model = Network1SDF.from_cfg(
        num_geometries=n_train_geoms,
        cfg=cfg
    ).to(device)

    # Lightning prefixes everything with "model."
    mapped = {k.replace("model.", ""): v for k, v in state.items() if k.startswith("model.")}

    model.load_state_dict(mapped, strict=False)
    model.eval()
    return model


# ------------------------------------------------------------
# Extract surface via marching cubes
# ------------------------------------------------------------
@torch.no_grad()
def extract_surface(model, geom_id, grid_res=160, bbox=0.65, device="cuda"):
    """
    geom_id : TRAIN geometry id (local id)
    bbox    : spatial extent [-bbox, bbox]^3
    """

    xs = np.linspace(-bbox, bbox, grid_res, dtype=np.float32)
    ys = np.linspace(-bbox, bbox, grid_res, dtype=np.float32)
    zs = np.linspace(-bbox, bbox, grid_res, dtype=np.float32)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    xyz = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    xyz_t = torch.from_numpy(xyz).to(device)
    gid_t = torch.full((xyz_t.shape[0],), geom_id, dtype=torch.long, device=device)

    # chunked inference (safe on GPU)
    sdf_vals = []
    chunk = 200_000
    for i in range(0, xyz_t.shape[0], chunk):
        sdf = model(gid_t[i:i+chunk], xyz_t[i:i+chunk])
        sdf_vals.append(sdf.squeeze(1).cpu().numpy())

    sdf_grid = np.concatenate(sdf_vals).reshape(grid_res, grid_res, grid_res)

    spacing = (xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0])
    verts, faces, normals, _ = measure.marching_cubes(
        sdf_grid, level=0.0, spacing=spacing
    )

    # shift vertices to world coordinates
    verts[:, 0] += xs[0]
    verts[:, 1] += ys[0]
    verts[:, 2] += zs[0]

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        process=False
    )
    return mesh


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    cfg = Network1Config()
    device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"

    ckpt_path = "/home/RUS_CIP/st184634/software_projects/atar_paper/checkpoints_network1/last.ckpt"   
    geom_id = 0                                    # TRAIN geometry id

    model = load_trained_model(ckpt_path, cfg, device)

    mesh = extract_surface(
        model,
        geom_id=geom_id,
        grid_res=160,   # increase for nicer mesh
        bbox=0.65,
        device=device
    )

    out_path = Path("network1_surface.ply")
    mesh.export(out_path)
    print(f"[OK] Surface saved to {out_path.resolve()}")

    mesh.show()


if __name__ == "__main__":
    main()
