import numpy as np
import utils
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

def remove_reflections(points, z_data_path, band_mm):
    valid_mask_1 = _remove_lower_edge_reflections(points, z_data_path, band_mm=band_mm)      
    valid_mask_2 = _remove_wall_reflections(points[valid_mask_1], z_data_path)      
    final_mask = np.zeros(len(points), dtype=bool)
    final_mask[valid_mask_1] = valid_mask_2 # combine masks
    return points[final_mask]

def _remove_lower_edge_reflections(points, z_data_path, resolution = 20000, band_mm=25.0, min_points=10):
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
    utils.visualise_removed_points(points, reflect_mask, resolution, plot_heading=f"Edge reflections removed from {z_data_path}")

    # Keep everything else
    keep_mask = ~reflect_mask
    return keep_mask

def __remove_wall_reflections(points, z_data_path, band_size=40,  dy=0.001, dz=0.001, min_points=1000, min_pts_per_cell=1, z_min_for_check=10, resolution=20000):
    Y, Z = points[:,1], points[:,2]
    y_max = np.max(Y)
    band_mask = (Y >= y_max - band_size)
    band = points[band_mask] # points in the band
   
    keep_mask = np.ones(points.shape[0], dtype=bool)
    if band.shape[0] < min_points:
        return keep_mask  # skip if the band is too sparse
    # z and y values in the band
    z_band = band[:, 2] 
    y_band = band[:, 1] 

    checkZ = (z_band >= z_min_for_check) # only consider points with sufficient Z

    # Determine number of bins in y and z
    y0, y1 = float(y_band.min()), float(y_band.max()) # Y range in the band
    z0, z1 = float(z_band.min()), float(z_band.max()) # Z range in the band
    ny = max(1, int(np.ceil((y1 - y0) / dy))) 
    nz = max(1, int(np.ceil((z1 - z0) / dz))) 

    occ = np.zeros((nz, ny), dtype=np.uint16) # occupancy grid
    # Populate occupancy grid
    iy = np.clip(((y_band - y0) / dy).astype(int), 0, ny - 1) # y bin indices
    iz = np.clip(((z_band - z0) / dz).astype(int), 0, nz - 1) # z bin indices
    np.add.at(occ, (iz[checkZ], iy[checkZ]), 1) # increment occupancy

    binary = occ >= min_pts_per_cell # discard tiny noise cells
    
    # find largest connected component
    labels, nlab = ndi.label(binary)
    if nlab <= 1:  # nothing or only one component found
        return keep_mask

    sizes = np.bincount(labels.ravel()) # component sizes
    sizes[0] = 0  # background label
    keep_lab = sizes.argmax() # label of the largest component
    main_blob = (labels == keep_lab) # mask of the largest component

    in_blob = main_blob[iz, iy]  # True if the (Yb,Zb) cell is in the largest component
    in_blob = (~checkZ) | in_blob # always keep low-Z points
    keep_mask[np.where(band_mask)[0]] = in_blob
        
    reflect_mask = ~keep_mask
    # Visualization of the reflection region
    utils.visualise_removed_points(points, reflect_mask, resolution, plot_heading=f"Wall reflections removed from {z_data_path}")

    return keep_mask



def _remove_wall_reflections(points, z_data_path, band_size=45, min_points=1000, z_thresh_min=2, z_thresh_max=30, resolution=20000, plot=True):
    """
    Removes reflection artifacts from vertical wall surfaces in the point cloud.

    parameters
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
    Y0_offset = 29.0 # offset for Y0 calculation; ie. the start of the slope line
    slope = -1.18 # slope of the threshold line in Z vs Y
    Y, Z = points[:,1], points[:,2]    
    y_max = float(np.max(Y))
    band_mask = (Y >= y_max - band_size)
    band_idx = np.where(band_mask)[0]
    band = points[band_mask] # points in the band

    keep_mask = np.ones(len(points), dtype=bool)

    if band.shape[0] < min_points: # skip if the band is too sparse
        if plot and band.size: # debug plot
            y_band = band[:,1]; z_band = band[:,2]
            Y0, Z0 = (y_max - Y0_offset), 3.0
            z_thresh = np.clip(Z0 + slope*(y_band - Y0), z_thresh_min, z_thresh_max)
            order = np.argsort(y_band)
            plt.figure(figsize=(7,6))
            plt.scatter(y_band, z_band, s=1, c='blue', alpha=0.5)
            plt.plot(y_band[order], z_thresh[order], 'g-', lw=2, label='Threshold')
            plt.xlabel("Y [mm]"); plt.ylabel("Z [mm]"); plt.title("Band too small; no filtering")
            plt.legend(); plt.tight_layout(); plt.show()
        return keep_mask

    # band coords
    y_band = band[:,1] # Y values in the band
    z_band = band[:,2] # Z values in the band

    # compute threshold line
    Y0, Z0 = (y_max - Y0_offset), 3.0 # starting point of the threshold line
    z_thresh = Z0 + slope * (y_band - Y0) # threshold Z values
    z_thresh = np.clip(z_thresh, z_thresh_min, z_thresh_max) # limit thresholds

    # keep BELOW the line, drop ABOVE the line 
    keep_in_band = (z_band <= z_thresh)
    keep_mask[band_idx] = keep_in_band

    # show what was removed 
    try:
        reflect_mask = ~keep_mask
        utils.visualise_removed_points(points, reflect_mask, resolution, plot_heading=f"Removed > threshold from {z_data_path}")
    except Exception:
        pass

    # debug plot
    if plot:
        plt.figure(figsize=(7,6))
        plt.scatter(y_band, z_band, s=1, c='blue', alpha=0.45, label='Band points')
        order = np.argsort(y_band)
        plt.plot(y_band[order], z_thresh[order], 'g-', lw=2.2, label='Threshold line')
        plt.xlabel("Y [mm]"); plt.ylabel("Z [mm]")
        plt.title("Keep below green line; remove above")
        plt.legend(); plt.tight_layout(); plt.show()

    return keep_mask