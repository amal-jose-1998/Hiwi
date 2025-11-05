import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def remove_reflections(points, band_mm):
    filtered_points, valid_mask = _remove_lower_edge_reflections(points, band_mm=band_mm)
    filtered_points, valid_mask = _remove_wall_reflections(filtered_points)
    return filtered_points, valid_mask

def _remove_lower_edge_reflections(points, resolution = 20000, band_mm=25.0, min_points=10):
    """
    Removes reflection spikes near the lowest Y edge of the point cloud.

    parameters
    points : ndarray of shape (N, 3)
        Input point cloud as (X, Y, Z) coordinates. 
    resolution : int
        Total number of points in the point cloud (used for subsampling in visualization).
    band_mm: float, optional, default=25.0
        how far from the minimum Y to consider (in mm)
    min_points: int, optional, default=10
        minimum number of points required to perform the operation  
    
    Returns
    filtered_points : ndarray of shape (M, 3)
        Point cloud with reflection artifacts removed.  
    keep_mask : ndarray of shape (N,), dtype=bool
        Boolean mask where True marks points that were kept.
    """

    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    y_min = np.min(Y)
    band_mask = (Y <= y_min + band_mm)

    band = points[band_mask] # points in the band
    keep_mask = np.ones(points.shape[0], dtype=bool)
    if band.shape[0] < min_points:
        return points, keep_mask  # skip if the band is too sparse
    
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
    visualise_removed_points(points, reflect_mask, resolution, plot_heading="red = lower edge reflections")

    # Keep everything else
    keep_mask = ~reflect_mask
    return points[keep_mask], keep_mask

def _remove_wall_reflections(points, standardize=True, resolution=20000):
    Y = points[:,1].astype(np.float64)
    Z = points[:,2].astype(np.float64)
    YZ = np.column_stack([Y, Z])

    if standardize:
        Y_median = np.median(YZ, axis=0)
        Y_std = np.std(YZ, axis=0) + 1e-9 # avoid division by zero
        YZ_std = (YZ - Y_median) / Y_std
    else:
        YZ_std = YZ
    
    keep_mask = np.ones(points.shape[0], dtype=bool)
    reflect_mask = ~keep_mask

    # Visualization of the reflection region
    visualise_removed_points(points, reflect_mask, resolution, plot_heading="red = wall reflections")
    return points[keep_mask], keep_mask


def visualise_removed_points(points, reflect_mask, resolution, plot_heading):
    color_mask = np.where(reflect_mask, 'r', 'b')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
     # Subsample
    plot_points = points
    plot_colors = color_mask
    if points.shape[0] > resolution:
        idx = np.random.choice(points.shape[0], resolution, replace=False)
        plot_points = points[idx]
        plot_colors = color_mask[idx]
    ax.scatter(plot_points[:,0], plot_points[:,1], plot_points[:,2], c=plot_colors, s=1)    
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(f"removed points ({plot_heading})")
    utils.maybe_show(fig)