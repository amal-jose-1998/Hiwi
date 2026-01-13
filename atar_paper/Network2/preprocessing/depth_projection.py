"""
Depth map generation from a triangle mesh.

This module converts an explicit surface mesh (V, F) into 2D depth maps:
    mesh (V,F) -> depth maps (X-direction and Z-direction)
    - X-projection map: image plane (Y, Z), pixel stores X coordinate (use depth_mode="max")
    - Z-projection map: image plane (X, Y), pixel stores Z coordinate (use depth_mode="max")
"""

import numpy as np

from atar_paper.Network2.config import DepthProjectionConfig


def _triangle_areas(V, F):
    """
    Compute per-triangle areas for a mesh.

    Parameters
    V :
        np.ndarray of shape (Nv, 3).
        Mesh vertices.
    F :
        np.ndarray of shape (Nf, 3).
        Triangle indices into V.

    Returns
    areas :
        np.ndarray of shape (Nf,), dtype float64.
        Area of each triangle.
    """
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return areas


def sample_surface_points(V, F, n_points, seed=0):
    """
    Sample points uniformly on the mesh surface (area-weighted triangle sampling).

    Parameters
    V :
        np.ndarray of shape (Nv, 3).
        Mesh vertices.
    F :
        np.ndarray of shape (Nf, 3).
        Triangle indices into V.
    n_points :
        int.
        Number of surface points to sample.
    seed :
        int.
        Random seed for reproducibility.

    Returns
    P :
        np.ndarray of shape (n_points, 3), dtype float32.
        Sampled points on the mesh surface.
    """
    V = np.asarray(V, dtype=np.float32)
    F = np.asarray(F, dtype=np.int64)

    if F.size == 0:
        raise ValueError("Mesh has no faces (F is empty).")

    areas = _triangle_areas(V, F).astype(np.float64)
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Invalid mesh: total surface area is zero or non-finite.")

    probs = areas / total

    rng = np.random.default_rng(int(seed))
    tri_idx = rng.choice(len(F), size=int(n_points), replace=True, p=probs)

    # Triangle vertices for chosen triangles
    v0 = V[F[tri_idx, 0]]
    v1 = V[F[tri_idx, 1]]
    v2 = V[F[tri_idx, 2]]

    # Uniform barycentric sampling:
    # r1, r2 ~ U[0,1]; u = 1 - sqrt(r1), v = sqrt(r1)*(1-r2), w = sqrt(r1)*r2
    r1 = rng.random(size=(n_points, 1), dtype=np.float32)
    r2 = rng.random(size=(n_points, 1), dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)

    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    P = u * v0 + v * v1 + w * v2
    return P.astype(np.float32)


def _auto_bounds_for_plane(points_2d, margin):
    """
    Compute square-ish bounds for a 2D plane based on point extents.

    Parameters
    points_2d :
        np.ndarray of shape (N, 2).
        2D projected coordinates.
    margin :
        float.
        Extra padding added on each side.

    Returns
    plane_min :
        np.ndarray of shape (2,), dtype float32.
    plane_max :
        np.ndarray of shape (2,), dtype float32.
    """
    mn = points_2d.min(axis=0)
    mx = points_2d.max(axis=0)

    # Make it symmetric-ish by using max span across the two axes.
    center = 0.5 * (mn + mx)
    span = float(np.max(mx - mn))
    half = 0.5 * span + float(margin)

    plane_min = (center - half).astype(np.float32)
    plane_max = (center + half).astype(np.float32)
    return plane_min, plane_max


def depth_map_from_points(points, axis, plane_axes, cfg, plane_min=None, plane_max=None):
    """
    Rasterize a depth map from a set of 3D points by binning onto a 2D grid.

    Parameters
    points :
        np.ndarray of shape (N, 3), dtype float32.
        3D points (typically sampled on mesh surface).
    axis :
        int in {0,1,2}.
        Viewing/depth axis index:
        - 0 => X depth
        - 1 => Y depth
        - 2 => Z depth
    plane_axes :
        tuple of two ints.
        Which axes form the 2D image plane (e.g., (1,2) for (Y,Z)).
    cfg :
        DepthProjectionConfig.
        Contains resolution, depth_mode, fill_value, margin, etc.
    plane_min :
        None or np.ndarray of shape (2,).
        Lower bounds (min) of the plane axes. If None, auto-computed from points.
    plane_max :
        None or np.ndarray of shape (2,).
        Upper bounds (max) of the plane axes. If None, auto-computed from points.

    Returns
    depth :
        np.ndarray of shape (H, W), dtype float32.
        Depth image. Empty pixels are filled with cfg.fill_value.
    mask :
        np.ndarray of shape (H, W), dtype uint8.
        1 where at least one point contributed, else 0.
    meta :
        dict.
        Contains plane bounds and pixel size:
        - "plane_min", "plane_max"
        - "pixel_size"
        - "depth_mode"
        - "axis", "plane_axes"
    """
    P = np.asarray(points, dtype=np.float32)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    H = int(cfg.resolution)
    W = int(cfg.resolution)
    if H <= 1 or W <= 1:
        raise ValueError("cfg.resolution must be >= 2.")

    pa0, pa1 = int(plane_axes[0]), int(plane_axes[1])

    plane = P[:, [pa0, pa1]]  # (N,2)
    depth_vals = P[:, int(axis)]  # (N,)

    if plane_min is None or plane_max is None:
        plane_min, plane_max = _auto_bounds_for_plane(plane, margin=float(cfg.margin))
    else:
        plane_min = np.asarray(plane_min, dtype=np.float32)
        plane_max = np.asarray(plane_max, dtype=np.float32)

    # Normalize to pixel coordinates
    eps = 1e-12
    span = np.maximum(plane_max - plane_min, eps)  # (2,)
    uv = (plane - plane_min[None, :]) / span[None, :]  # [0,1] ideally

    # Convert to pixel indices
    # u -> x pixel (col), v -> y pixel (row)
    col = np.floor(uv[:, 0] * (W - 1)).astype(np.int64)
    row = np.floor(uv[:, 1] * (H - 1)).astype(np.int64)

    # Filter in-bounds
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    row = row[ok]
    col = col[ok]
    d = depth_vals[ok].astype(np.float32)

    fill = float(cfg.fill_value)
    depth = np.full((H, W), fill, dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)

    mode = str(cfg.depth_mode).lower().strip()
    if mode not in ("min", "max"):
        raise ValueError("cfg.depth_mode must be 'min' or 'max'.")

    # Aggregate depth per pixel. Use in-place min/max.
    # We do a simple loop for clarity; resolution is modest and points are sampled.
    for r, c, dv in zip(row, col, d):
        if mask[r, c] == 0:
            depth[r, c] = dv
            mask[r, c] = 1
        else:
            if mode == "min":
                if dv < depth[r, c]:
                    depth[r, c] = dv
            else:  # mode == "max"
                if dv > depth[r, c]:
                    depth[r, c] = dv

    pixel_size = (span / np.array([W - 1, H - 1], dtype=np.float32)).astype(np.float32)

    meta = {
        "plane_min": plane_min.astype(np.float32),
        "plane_max": plane_max.astype(np.float32),
        "pixel_size": pixel_size,
        "depth_mode": mode,
        "axis": int(axis),
        "plane_axes": (pa0, pa1),
    }
    return depth, mask, meta


def project_mesh_to_depth_maps(V, F, cfg=None, seed=0):
    """
    Generate the two depth maps required by Network-2 from a triangle mesh.

    This produces:
    - depth_x: projection along X onto the (Y,Z) plane
    - depth_z: projection along Z onto the (X,Y) plane

    Parameters
    V :
        np.ndarray of shape (Nv, 3).
        Mesh vertices in world coordinates.
    F :
        np.ndarray of shape (Nf, 3).
        Triangle indices into V.
    cfg :
        DepthProjectionConfig or None.
        If None, DepthProjectionConfig() defaults are used.
    seed :
        int.
        Random seed for surface point sampling.

    Returns
    depth_x :
        np.ndarray of shape (H, W), dtype float32.
        Depth map for X-direction projection (image plane YZ, depth=X).
    depth_z :
        np.ndarray of shape (H, W), dtype float32.
        Depth map for Z-direction projection (image plane XY, depth=Z).
    mask_x :
        np.ndarray of shape (H, W), dtype uint8.
        Valid mask for depth_x.
    mask_z :
        np.ndarray of shape (H, W), dtype uint8.
        Valid mask for depth_z.
    meta :
        dict.
        Metadata for both projections, including:
        - sampling info (n_surface_points)
        - per-map plane bounds and pixel sizes
    """
    cfg = DepthProjectionConfig() if cfg is None else cfg

    V = np.asarray(V, dtype=np.float32)
    F = np.asarray(F, dtype=np.int64)

    # Sample points on mesh surface
    P = sample_surface_points(V, F, n_points=int(cfg.n_surface_points), seed=int(seed))

    # X-direction: depth = X, plane = (Y,Z)
    depth_x, mask_x, meta_x = depth_map_from_points(
        points=P,
        axis=0,
        plane_axes=(1, 2),
        cfg=cfg,
        plane_min=cfg.plane_min_yz,
        plane_max=cfg.plane_max_yz,
    )

    # Z-direction: depth = Z, plane = (X,Y)
    depth_z, mask_z, meta_z = depth_map_from_points(
        points=P,
        axis=2,
        plane_axes=(0, 1),
        cfg=cfg,
        plane_min=cfg.plane_min_xy,
        plane_max=cfg.plane_max_xy,
    )

    meta = {
        "n_surface_points": int(cfg.n_surface_points),
        "depth_x": meta_x,
        "depth_z": meta_z,
    }
    return depth_x, depth_z, mask_x, mask_z, meta
