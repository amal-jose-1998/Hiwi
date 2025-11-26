import numpy as np
import utils
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from outlier_correction import OutlierRemover


def remove_reflections(op_flag, points, z_data_path, band_mm, visualise):
    valid_mask_1 = _remove_lower_edge_reflections(op_flag, points, z_data_path, band_mm=band_mm, visualise=visualise)      
    #valid_mask_2 = remove_wall_reflections(points[valid_mask_1], z_data_path, visualise=visualise) 
    #valid_mask_2 = _remove_wall_reflections(op_flag, points[valid_mask_1], z_data_path, visualise=visualise)     
    valid_mask_2 = __remove_wall_reflections(op_flag, points[valid_mask_1], z_data_path, visualise=visualise) 
    final_mask = np.zeros(len(points), dtype=bool)
    final_mask[valid_mask_1] = valid_mask_2 # combine masks
    return points[final_mask]


def _remove_lower_edge_reflections(op_flag, points, z_data_path, resolution = 20000, band_mm=25.0, min_points=10, visualise=True):
    """
    Removes reflection spikes near the lowest Y edge of the point cloud.

    parameters
    op_flag : str
        Operation flag indicating the type of operation (e.g., "OP_10" or "OP_20").
    points : ndarray of shape (N, 3)
        Input point cloud as (X, Y, Z) coordinates. 
    resolution : int
        Total number of points in the point cloud (used for subsampling in visualization).
    band_mm: float, optional, default=25.0
        how far from the minimum Y to consider (in mm)
    min_points: int, optional, default=10
        minimum number of points required to perform the operation  
    
    Returns
    keep_mask : ndarray of shape (N,), dtype=bool
        Boolean mask where True marks points that were kept.
    """

    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    y_min = np.min(Y)
    y_max = np.max(Y)
    x_min = np.min(X)
    x_max = np.max(X)

    band_mask_1 = (Y <= y_min + band_mm)
    band_mask_2 = (Y >= y_max - band_mm) 
    band_mask_3 = (X <= x_min + band_mm)
    band_mask_4 = (X >= x_max - band_mm)
    band_mask = band_mask_1 | band_mask_2 | band_mask_3 | band_mask_4
    
    if op_flag == "OP_10":
        band_mask = band_mask_1
        print("band_mask_1 applied for OP_10")
    
    band = points[band_mask] # points in the band
    keep_mask = np.ones(points.shape[0], dtype=bool)
    if band.shape[0] < min_points:
        return keep_mask  # skip if the band is too sparse
    
    z_band = band[:, 2] # Z values in the band
    z_med = np.median(z_band) # median Z in the band
    mad = np.median(np.abs(z_band - z_med)) # median absolute deviation in the band. This gives a robust measure of spread.
    # Convert MAD to robust sigma (Gaussian equiv.)
    robust_sigma = 1.4826 * mad if mad > 0 else np.std(z_band) # fallback to std if mad is zero
    k_sigma = 2.4  # threshold factor
    # Baseline Z above which points are considered reflections
    z_baseline = z_med + k_sigma * robust_sigma
    
    # Points to drop: in the band AND well above the baseline
    reflect_mask = band_mask & (Z > (z_baseline))

    # Visualization of the reflection region
    if visualise:
        utils.visualise_removed_points(points, reflect_mask, resolution, plot_heading=f"Edge reflections removed from {z_data_path}")

    # Keep everything else
    keep_mask = ~reflect_mask
    return keep_mask

def _remove_wall_reflections(op_flag, points, z_data_path, band_size=48, min_points=1000, z_thresh_min=3.5, z_thresh_max=30, resolution=20000, plot=True, visualise=True):
    """
    Removes reflection artifacts from vertical wall surfaces in the point cloud, using a dynamic threshold line in Z vs Y defined by a preset slope and offset.

    parameters
    op_flag : str
        Operation flag indicating the type of operation (e.g., "OP_10" or "OP_20").
    points : ndarray of shape (N, 3)
        Input point cloud as (X, Y, Z) coordinates.
    z_data_path : str
        Path to the original z_data file (used for plot titles).
    band_size : float, optional, default=45
        Size of the band from the maximum Y to consider (in mm).
    min_points : int, optional, default=1000
        Minimum number of points required in the band to perform reflection removal.
    z_thresh_min : float, optional, default=2
        Minimum Z threshold for reflection removal.
    z_thresh_max : float, optional, default=30
        Maximum Z threshold for reflection removal
    resolution : int, optional, default=20000
        Total number of points in the point cloud (used for subsampling in visualization).
    plot : bool, optional, default=True
        Whether to generate debug plots.
    
    Returns
    keep_mask : ndarray of shape (N,), dtype=bool
        Boolean mask where True marks points that were kept.
    """
    if op_flag == "OP_10":
        Y0_offset = 24# offset for Y0 calculation; ie. y coordinate of the start of the slope line
        slope = -1.44 # slope of the threshold line in Z vs Y
    else: # OP_20
        Y0_offset = 14# offset for Y0 calculation; ie. y coordinate of the start of the slope line
        slope = -2.0 # slope of the threshold line in Z vs Y
    Y, Z = points[:,1], points[:,2]    
    y_max = float(np.max(Y))
    band_mask = (Y >= y_max - band_size)
    band_idx = np.where(band_mask)[0]
    band = points[band_mask] # points in the band

    keep_mask = np.ones(len(points), dtype=bool)

    if band.shape[0] < min_points: # skip if the band is too sparse
        return keep_mask

    # band coords
    y_band = band[:,1] # Y values in the band
    z_band = band[:,2] # Z values in the band

    # compute threshold line
    Y0, Z0 = (y_max - Y0_offset), 4 # starting point of the threshold line
    z_thresh = Z0 + slope * (y_band - Y0) # threshold Z values
    z_thresh = np.clip(z_thresh, z_thresh_min, z_thresh_max) # limit thresholds

    # keep BELOW the line, drop ABOVE the line 
    keep_in_band = (z_band <= z_thresh)
    keep_mask[band_idx] = keep_in_band

    if visualise:
        # show what was removed 
        try:
            reflect_mask = ~keep_mask
            utils.visualise_removed_points(points, reflect_mask, resolution, plot_heading=f"Removed > threshold from {z_data_path}")
        except Exception:
            pass

        # debug plot
        if plot:
            utils.visualise_boundary_line(y_band, z_thresh, z_band)
        
    return keep_mask


def knn_cache(points, k=20, batch=None):
    """
    Build a cKDTree once, return kNN distances/indices for every point.
    If 'batch' is set (int), queries in chunks to reduce peak memory.
    """
    tree = cKDTree(points)
    N = len(points)
    dists = np.empty((N, k), dtype=np.float32)
    idx   = np.empty((N, k), dtype=np.int32)

    if batch is None:
        d, i = tree.query(points, k=k)
        dists[:] = d
        idx[:]   = i
    else:
        for s in range(0, N, batch):
            e = min(N, s+batch)
            d, i = tree.query(points[s:e], k=k)
            dists[s:e] = d
            idx[s:e]   = i
    return dists, idx

def density_score_from_knn(dists):
    # exclude self at col 0
    mean_knn = dists[:, 1:].mean(axis=1)
    return 1.0 / (mean_knn + 1e-9)  # higher = denser (better)

def spacing_cv(dists):
    dd = dists[:, 1:]
    mu = dd.mean(axis=1) + 1e-9
    sd = dd.std(axis=1)
    return sd / mu  # higher = irregular (worse)

def kth_distance_stability(dists):
    kth = dists[:, -1]
    med = np.median(kth)
    mad = 1.4826 * np.median(np.abs(kth - med)) + 1e-9
    z = np.abs((kth - med) / mad)
    return -z  # higher = more typical neighborhood (better)

def surface_variation(points, idx):
    """
    λ3 / (λ1+λ2+λ3) using local covariance on kNN.
    Higher = scattered (worse). Uses a simple Python loop; numba can JIT it.
    """
    N = len(points)
    sv = np.zeros(N, dtype=np.float32)
    for i in range(N):
        neigh = points[idx[i]]
        C = np.cov(neigh.T)
        w = np.linalg.eigvalsh(C)  # ascending: λ1<=λ2<=λ3
        s = w.sum()
        if s > 0:
            sv[i] = w[0] / s  # smallest / total
    return sv

def robust_unit(score, higher_is_better=True):
    med = np.median(score)
    mad = 1.4826 * np.median(np.abs(score - med)) + 1e-9
    z = (score - med) / mad
    # squash to [0,1] for stability
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -6, 6)))
    return s if higher_is_better else 1 - s

def combine_scores(scores, weights=None):
    if weights is None:
        weights = [1.0] * len(scores)
    S = np.zeros_like(scores[0], dtype=np.float32)
    wsum = float(np.sum(weights)) + 1e-9
    for w, sc in zip(weights, scores):
        S += w * sc
    return S / wsum

def sor_mask(points, k=18, zmax=3.0):
    """
    Statistical Outlier Removal on mean k-NN distance (robust z-score).
    Keep points whose mean neighbor distance is within zmax MADs.
    """
    tree = cKDTree(points.astype(np.float32, copy=False))
    dists, _ = tree.query(points, k=k)  # dists[:,0] == 0
    mu = dists[:, 1:].mean(axis=1)

    med = np.median(mu)
    mad = 1.4826 * np.median(np.abs(mu - med)) + 1e-9
    z = (mu - med) / mad
    return z <= zmax

def remove_wall_reflections(points, z_data_path, k_neighbors=20, drop_frac=0.05, weights=(1.0, 1.0, 0.7, 0.5), planar_q=0.95, sor_k=18, sor_zmax=3.2, batch_knn=None, visualise=True):
    """
    Returns keep_mask.
    weights map to: [density, (1-surface_variation), (1-spacing_cv), kth_stability]
    """
    # ensure float32
    pts = np.asarray(points, dtype=np.float32)
    N   = len(pts)

    # one kNN for all
    dists, idx = knn_cache(pts, k=k_neighbors, batch=batch_knn)

    # raw scores
    dens = density_score_from_knn(dists)      # higher better
    svar = surface_variation(pts, idx)        # higher worse
    cv   = spacing_cv(dists)                  # higher worse
    kstab= kth_distance_stability(dists)      # higher better

    # normalize with correct polarity
    s1 = robust_unit(dens, higher_is_better=True)
    s2 = robust_unit(svar, higher_is_better=False)
    s3 = robust_unit(cv,   higher_is_better=False)
    s4 = robust_unit(kstab,higher_is_better=True)

    combo = combine_scores([s1, s2, s3, s4], weights=weights)
    thr = np.quantile(combo, drop_frac)
    keep = combo >= thr

    # ---- Planarity safeguard: preserve clearly planar patches (sidewalls)
    # lower svar = more planar; keep the lowest 'planar_q' quantile
    svar_thr    = np.quantile(svar, planar_q)
    planar_keep = (svar <= svar_thr)

    # ---- SOR only on risky subset: kernel-kept & NOT planar
    cand = keep & (~planar_keep)
    sor_keep_sub = np.zeros(cand.sum(), dtype=bool)
    if cand.any():
        sor_keep_sub = sor_mask(pts[cand], k=sor_k, zmax=sor_zmax)

    # ---- assemble final mask
    final_mask = np.zeros(N, dtype=bool)
    final_mask[planar_keep] = True          # keep all planar patches
    final_mask[cand]        = sor_keep_sub  # keep non-planar only if SOR says so
    # (points not in keep_kernel remain False)

    # ---- visualize ALL wall-stage removals (kernel rejects + SOR rejects)
    removed_mask = ~final_mask

    if visualise:
        try:
            utils.visualise_removed_points(points, removed_mask, resolution=20000, plot_heading=f"Wall Reflections removed from {z_data_path}")
        except Exception:
            pass
    return final_mask

def __remove_wall_reflections(
    op_flag,
    points,
    z_data_path,
    band_size_use=48,
    min_points=1000,
    dy_mm=5.0,
    min_pts_per_bin=400,
    k_sigma=1.6,
    z_margin=0.5,
    z_thresh_min=3.5,
    z_thresh_max=30.0,
    resolution=20000,
    plot=True,
    visualise=True,
):
    """
    Remove wall-reflection spikes using a *local* Z threshold per Y-slice
    instead of a single global line.

    Steps:
    - Take a Y-band near y_max (size = band_size).
    - Discretise that band into bins of thickness dy_mm.
    - In each bin:
        * Compute median Z and robust sigma (from MAD).
        * Threshold = median + k_sigma * sigma + z_margin.
    - Clip thresholds to [z_thresh_min, z_thresh_max].
    - Points in the band above their local threshold are removed.
    """

    pts = np.asarray(points)
    Y = pts[:, 1]
    Z = pts[:, 2]

    # you can still tweak band_size per op if you want
    if op_flag == "OP_10":
        k_sigma=1.9
        dy_mm=4.5
    else:
        k_sigma=1.6
        dy_mm=5.0

    y_max = float(Y.max())
    band_mask = (Y >= y_max - band_size_use)
    band_idx = np.where(band_mask)[0]
    band = pts[band_mask]

    keep_mask = np.ones(len(pts), dtype=bool)

    if band.shape[0] < min_points:
        # Too sparse: don't try anything fancy
        return keep_mask

    y_band = band[:, 1]
    z_band = band[:, 2]

    # ---- build local thresholds per Y-bin ----
    y_min_band = float(y_band.min())
    y_max_band = float(y_band.max())

    if not np.isfinite(y_min_band) or not np.isfinite(y_max_band) or y_max_band <= y_min_band:
        return keep_mask

    # number of bins from band width / dy_mm (clamped for sanity)
    n_bins = max(4, int(np.round((y_max_band - y_min_band) / dy_mm)))
    edges = np.linspace(y_min_band, y_max_band, n_bins + 1)

    # one threshold per point in the band (initially "unset" = NaN)
    z_thresh_per_point = np.full_like(z_band, np.nan, dtype=float)

    for i in range(n_bins):
        sel = (y_band >= edges[i]) & (y_band < edges[i + 1])
        n_sel = int(np.count_nonzero(sel))
        if n_sel < min_pts_per_bin:
            continue  # not enough data in this slice

        z_slice = z_band[sel]

        z_med = np.median(z_slice)
        mad = np.median(np.abs(z_slice - z_med))
        if mad > 0:
            sigma = 1.4826 * mad
        else:
            sigma = np.std(z_slice)

        # local threshold for this Y range
        z_thr = z_med + k_sigma * sigma + z_margin
        z_thr = float(np.clip(z_thr, z_thresh_min, z_thresh_max))

        z_thresh_per_point[sel] = z_thr

    # points with a defined threshold: keep if below, remove if above
    has_thr = np.isfinite(z_thresh_per_point)
    keep_in_band = np.ones_like(z_band, dtype=bool)
    keep_in_band[has_thr] = (z_band[has_thr] <= z_thresh_per_point[has_thr])

    # write back into global mask
    keep_mask[band_idx] = keep_in_band

    if visualise:
        # show what was removed
        try:
            removed_mask = ~keep_mask
            utils.visualise_removed_points(
                points,
                removed_mask,
                resolution,
                plot_heading=f"Wall reflections removed (local Y-slices) from {z_data_path}",
            )
        except Exception:
            pass

        # optional Y–Z debug plot of boundary curve
        if plot:
            try:
                # only plot where threshold exists
                y_plot = y_band[has_thr]
                z_thr_plot = z_thresh_per_point[has_thr]
                utils.visualise_boundary_line(y_plot, z_thr_plot, z_band)
            except Exception:
                pass

    return keep_mask
