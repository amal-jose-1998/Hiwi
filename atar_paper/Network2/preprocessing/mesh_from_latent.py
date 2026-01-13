"""
Network-2 preprocessing: mesh extraction from a Network-1 latent vector.

Forward path:
    z  --(Network-1 SDF decoder)-->  SDF scalar field on a 3D grid
      --(Marching Cubes @ iso=0)-->  mesh (V, F)
"""

from atar_paper.Network2.config import MeshFromLatentConfig

import numpy as np
import torch
from skimage.measure import marching_cubes


def _make_xyz_grid(cfg):
    """
    Create a dense 3D grid of query points.

    Parameters
    cfg :
        MeshFromLatentConfig.
        Uses cfg.grid_res, cfg.grid_min, cfg.grid_max.

    Returns
    xyz :
        np.ndarray of shape (N^3, 3), dtype float32.
        Flattened query points in world coordinates.
    spacing :
        tuple of 3 floats (sx, sy, sz).
        Voxel spacing in world units. Used by Marching Cubes.
    origin :
        tuple of 3 floats (ox, oy, oz).
        World coordinate corresponding to voxel index (0, 0, 0).
    """
    N = int(cfg.grid_res)
    gmin = float(cfg.grid_min)
    gmax = float(cfg.grid_max)
    if gmax <= gmin:
        raise ValueError("grid_max must be > grid_min.")

    # linspace includes endpoints; spacing based on N-1 intervals
    xs = np.linspace(gmin, gmax, N, dtype=np.float32)
    ys = np.linspace(gmin, gmax, N, dtype=np.float32)
    zs = np.linspace(gmin, gmax, N, dtype=np.float32)

    # Create grid in world coords
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")  # each (N,N,N)
    xyz = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    # spacing for converting marching-cubes vertices back to world units
    if N <= 1:
        raise ValueError("grid_res must be >= 2.")
    s = (gmax - gmin) / float(N - 1)
    spacing = (s, s, s)
    origin = (gmin, gmin, gmin)
    return xyz, spacing, origin


@torch.no_grad()
def sdf_grid_from_latent(decoder, z, cfg):
    """
    Evaluate the Network-1 decoder on a dense 3D grid to produce an SDF volume.

    Parameters
    decoder :
        Trained Network-1 decoder module.
        Must implement: decoder.forward_with_latent(z, xyz) -> sdf.
    z :
        torch.Tensor.
        Latent code for a single geometry.
        Expected shape: (L,) or (1, L). If (1, L), it will be squeezed to (L,).
    cfg :
        MeshFromLatentConfig.
        Controls grid creation, chunking, device and dtype.

    Returns
    sdf_vol :
        np.ndarray of shape (N, N, N), dtype float32.
        Dense SDF values on the grid. Axis order matches grid creation: (x, y, z).
    meta :
        dict.
        Metadata describing the grid and inference, with keys:
        - "grid_res", "grid_min", "grid_max"
        - "origin"  : (ox, oy, oz)
        - "spacing" : (sx, sy, sz)
        - "iso_level"
        - "device_used"
    """
    # Device
    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    decoder = decoder.to(device)
    decoder.eval()

    # Prepare grid
    xyz_np, spacing, origin = _make_xyz_grid(cfg)
    N = int(cfg.grid_res)

    # Prepare latent
    if not torch.is_tensor(z):
        raise TypeError("z must be a torch.Tensor")
    z = z.to(device=device, dtype=cfg.dtype)
    if z.dim() == 2 and z.shape[0] == 1:
        z = z.squeeze(0)
    if z.dim() != 1:
        raise ValueError(f"z must be 1D (L,) or (1,L). Got shape={tuple(z.shape)}")

    # Chunked inference
    sdf_out = np.empty((xyz_np.shape[0],), dtype=np.float32)
    chunk = int(cfg.chunk_size)

    for start in range(0, xyz_np.shape[0], chunk):
        end = min(start + chunk, xyz_np.shape[0])
        xyz_chunk = torch.from_numpy(xyz_np[start:end]).to(device=device, dtype=cfg.dtype)  # (B,3)

        # decoder API: forward_with_latent(z, xyz) -> (B,1)
        pred = decoder.forward_with_latent(z, xyz_chunk)
        pred = pred.reshape(-1).detach().to("cpu").numpy().astype(np.float32)
        sdf_out[start:end] = pred

    sdf_vol = sdf_out.reshape(N, N, N)

    meta = {
        "grid_res": N,
        "grid_min": float(cfg.grid_min),
        "grid_max": float(cfg.grid_max),
        "origin": origin,
        "spacing": spacing,
        "iso_level": float(cfg.iso_level),
        "device_used": str(device),
    }
    return sdf_vol, meta


def mesh_from_latent(decoder, z, cfg=None):
    """
    Extract a triangle surface mesh from a latent geometry code.
    Full pipeline: z -> SDF volume -> Marching Cubes -> mesh.

    Parameters
    decoder :
        Trained Network-1 decoder module.
        Must implement: decoder.forward_with_latent(z, xyz) -> sdf.
    z :
        torch.Tensor.
        Latent code for a single geometry. Expected shape: (L,) or (1, L).
    cfg :
        MeshFromLatentConfig or None.
        If None, MeshFromLatentConfig() defaults are used.

    Returns
    V :
        np.ndarray of shape (Nv, 3), dtype float32.
        Mesh vertices in world coordinates.
    F :
        np.ndarray of shape (Nf, 3), dtype int64.
        Triangle indices into V.
    meta :
        dict.
        Grid metadata plus diagnostics:
        - "n_verts", "n_faces"
        - "sdf_min", "sdf_max"
        plus the keys returned by sdf_grid_from_latent().
    """
    cfg = MeshFromLatentConfig() if cfg is None else cfg

    sdf_vol, meta = sdf_grid_from_latent(decoder=decoder, z=z, cfg=cfg)
    spacing = tuple(meta["spacing"])
    origin = np.array(meta["origin"], dtype=np.float32)

    verts, faces, normals, values = marching_cubes(
        volume=sdf_vol,
        level=float(cfg.iso_level),
        spacing=spacing,   # converts index space -> world increments
    )

    # marching_cubes with spacing gives verts in world units relative to origin at (0,0,0) index.
    # We need to shift by origin.
    V = (verts.astype(np.float32) + origin[None, :]).astype(np.float32)
    F = faces.astype(np.int64)

    meta_out = dict(meta)
    meta_out.update(
        {
            "n_verts": int(V.shape[0]),
            "n_faces": int(F.shape[0]),
            "sdf_min": float(np.min(sdf_vol)),
            "sdf_max": float(np.max(sdf_vol)),
        }
    )
    return V, F, meta_out


def save_mesh_as_ply(path, V, F):
    """
    Minimal PLY writer (triangle mesh). Useful for quick debugging/visualization.

    Parameters
    path :
        str.
        Output filepath ending with ".ply".
    V :
        np.ndarray of shape (Nv, 3).
        Mesh vertices in world coordinates. Will be cast to float32.
    F :
        np.ndarray of shape (Nf, 3).
        Triangle face indices into V. Will be cast to int64.

    """
    V = np.asarray(V, dtype=np.float32)
    F = np.asarray(F, dtype=np.int64)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {V.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {F.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in V:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for tri in F:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
