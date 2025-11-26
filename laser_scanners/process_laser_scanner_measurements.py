import numpy as np
import matplotlib.pyplot as plt
from outlier_correction import OutlierRemover
from scale_correction import Callibrator
from reflection_remover import remove_reflections
from glob import glob
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
import utils

base_paths = glob("/mnt/nas/uncompressed_data/*/Keyence-Messung/")

# Laser Sensor: 
# - 3200 points per line
# - 1600 lines in total
width = 3200
length = 1600

resolution = 20000 # Number of points to random sample. Increased=more details, Decreased=faster visualization.
remove_invalid = True # If True remove points, that are invalid (like double.nan)
scaling_factor_in_Z = 30 # Scale Z to this value (mm)
visualise = False # If True, visualise the detected bounding box, the workpiece, removed outliers from the point cloud and the final point cloud

def visualize_band_boundary(points, y_min, band_mm, resolution=20000, title="Band boundary"):
    """
    Shows a horizontal boundary line at Y = y_min + band_mm on the point cloud.
    """
    import matplotlib.pyplot as plt

    pts = np.asarray(points)
    X = pts[:, 0]
    Y = pts[:, 1]
    Z = pts[:, 2]

    # Subsample for speed
    if len(points) > resolution:
        idx = np.random.choice(len(points), resolution, replace=False)
        X, Y, Z = X[idx], Y[idx], Z[idx]

    boundary_y = y_min + band_mm

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=1)

    # Draw boundary line across entire X-range
    x_line = np.array([X.min(), X.max()])
    y_line = np.array([boundary_y, boundary_y])
    z_line = np.array([Z.min(), Z.min()])  # flat line in Z just for visualization

    ax.plot(x_line, y_line, z_line, color='red', linewidth=3)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(title)

    utils.maybe_show(fig)

def _estimate_lower_edge_band_mm(points, max_band_mm=25, dy_mm=1.0, z_floor_quantile=0.1, z_height_quantile=0.9, z_thresh=5.0, min_points_per_slice=5000):
    """
    Estimate a reflection-removal band along the lower Y-edge.

    points : (N, 3) in [X, Y, Z] mm (after scaling)
    max_band_mm : maximum band size to search [mm]
    dy_mm : slice thickness in Y [mm]
    z_floor_quantile : quantile to estimate floor height (low Z)
    z_height_quantile: quantile to detect tall structures (walls/legs)
    z_thresh : additional Z above floor to consider 'wall' present [mm]
    """
    pts = np.asarray(points)
    Y = pts[:, 1]
    Z = pts[:, 2]
    y_min = float(Y.min())
    y_max_search = y_min + max_band_mm
    # Estimate global 'floor' height from low Z values
    z_floor = float(np.quantile(Z, z_floor_quantile))
    band_mm = 0.0
    y = y_min
    while y < y_max_search:
        slice_mask = (Y >= y) & (Y < y + dy_mm)
        if np.count_nonzero(slice_mask) < min_points_per_slice:
            y += dy_mm
            continue
        z_slice = Z[slice_mask]
        z_hi = float(np.quantile(z_slice, z_height_quantile))
        # If high quantile is still near the floor -> still in reflection strip
        if z_hi < z_floor + z_thresh:
            band_mm = (y + dy_mm) - y_min  # extend band down to this slice
            y += dy_mm
        else:
            # We've hit real structure (legs/walls); stop band before this
            break

    # Safety clamp
    band_mm = max(0.0, min(band_mm, max_band_mm))
    if visualise and band_mm > 0:
        visualize_band_boundary(points, y_min, band_mm, resolution=resolution, title=f"Lower-edge band boundary ({band_mm+y_min:.1f} mm)")
    return band_mm

def apply_scaling(X_px, Y_px, Z, sx, sy, scaling_factor_in_Z):
    """Apply scaling to pixel coordinates and Z values."""
    X = X_px * sx
    Y = Y_px * sy

    z_min = float(Z.min())
    z_max = float(Z.max())
    den   = max(z_max - z_min, 1e-9)
    Z_scaled  = (Z - z_min) / den * scaling_factor_in_Z
    return np.column_stack((X, Y, Z_scaled))

def run_my_logic(op_tag, outlier_remover, length, width, sx, sy, bbox = None, z_data_path = "/home/RUS_CIP/st184634/software_projects/laser_scanners/pc_measurement/z_data.npy", idx_str="0000"):
    if visualise:
        utils.plot_original_point_cloud(length, width, z_data_path, resolution, title = z_data_path)
    z = load_z_2d(z_data_path, width_hint=width, length_hint=length) 
    y, x = np.indices(z.shape) 
    # Prepare XYZ from kept pixels
    X_px_all = x.ravel()
    Y_px_all = y.ravel()
    Z_all = z.ravel()
    original_points = apply_scaling(X_px_all, Y_px_all, Z_all, sx, sy, scaling_factor_in_Z) #scaled original points before outlier removal
    # Remove outliers
    outlier_removed = None
    if op_tag == "OP_10":
        mask = outlier_remover.remove_outliers(z) # mask for real measurement with outliers removed
        outlier_removed = ~mask.ravel()
        X_px, Y_px, Z = x[mask], y[mask], z[mask]
        points = apply_scaling(X_px, Y_px, Z, sx, sy, scaling_factor_in_Z) # Apply scaling
        if visualise:
            utils.visualise_removed_points(original_points, outlier_removed, resolution, plot_heading=f"Outliers removed from {z_data_path}")
        band_mm = _estimate_lower_edge_band_mm(points)
        band_mm-=2.5
        print(f"{op_tag}: estimated lower-edge band = {band_mm:.2f} mm for {z_data_path}")
    else:
        valid_mask = outlier_remover.remove_invalid_z(z) # mask for real measurement with outliers removed
        outlier_removed = ~valid_mask.ravel()
        if visualise:
            utils.visualise_removed_points(original_points, outlier_removed, resolution, plot_heading=f"Outliers removed from {z_data_path}")
        valid_gradient_mask = outlier_remover.gradient_z_filter(z) # removes side walls
        valid_core_mask = valid_mask & valid_gradient_mask        # point cloud without sidewalls used for O3D cleanup
        wall_mask = valid_mask & ~valid_gradient_mask       # points of side walls only
        X_core, Y_core, Z_core = x[valid_core_mask], y[valid_core_mask], z[valid_core_mask]
        X_wall, Y_wall, Z_wall = x[wall_mask], y[wall_mask], z[wall_mask]
        points_core = apply_scaling(X_core, Y_core, Z_core, sx, sy, scaling_factor_in_Z)
        points_wall = apply_scaling(X_wall, Y_wall, Z_wall, sx, sy, scaling_factor_in_Z)
        points_core_clean = outlier_remover.o3d_statistical_cleanup(points_core, z_data_path, visualise=visualise, resolution=resolution)
        points = np.vstack([points_core_clean, points_wall])
        points = outlier_remover.local_planarity_filter(points, k=20, curvature_threshold=0.02, visualise=visualise, resolution=resolution, title=f"Not planar points removed from {z_data_path}")
        band_mm = _estimate_lower_edge_band_mm( points, max_band_mm=17, z_thresh = 2.0, z_height_quantile = 0.95, min_points_per_slice=2000)
        band_mm-=2.5
        print(f"{op_tag}: estimated lower-edge band = {band_mm:.2f} mm for {z_data_path}")

    refined_points = remove_reflections(op_flag=op_tag, points=points, z_data_path=z_data_path, band_mm=band_mm, visualise=visualise)
    
    if refined_points is None or refined_points.size == 0:
        raise ValueError(f"No points left after reflection removal for {z_data_path}.")

    save_dir = Path("/home/RUS_CIP/st184634/software_projects/laser_scanners/refined_data/")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"{idx_str}_{op_tag}.npy"
    save_path = save_dir / save_name
    np.save(save_path, refined_points)
    print(f"Saved: {save_path}")

    if visualise:
        if refined_points.shape[0] > resolution:
            idx = np.random.choice(refined_points.shape[0], resolution, replace=False)
            refined_points = refined_points[idx]
        # Visualize final point cloud
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(refined_points[:,0], refined_points[:,1], refined_points[:,2], c=refined_points[:,2], cmap='viridis', s=1)

        ax.set_xlabel("X [mm]" if bbox is not None else "X [px]")
        ax.set_ylabel("Y [mm]" if bbox is not None else "Y [px]")
        ax.set_zlabel("Z [mm]")
        ax.set_xlim([0,200])
        ax.set_ylim([0,200])
        ax.set_zlim([0,scaling_factor_in_Z])
        ax.set_title(f"Processed Point Cloud from {z_data_path}", fontsize=9, wrap=True)
        utils.maybe_show(fig)

def quick_get_pairs(main_folder_path: str) -> List[Tuple[str, str]]:
    """Compact version for quick access to pairs"""
    p = Path(main_folder_path)
    dirs = sorted(p.iterdir(), key=lambda x: int(x.name.split('_')[0]))
    return [(str(dirs[i]/"OP_10"), str(dirs[i+2]/"OP_20")) 
            for i in range(len(dirs)-1) 
            if (dirs[i]/"OP_10").exists() and (dirs[i+2]/"OP_20").exists()]

def load_z_2d(path, width_hint=None, length_hint=None):
    """Load .npy and return a 2D array shaped (length, width)."""
    a = np.load(path)
    # Already 2D?
    if a.ndim == 2:
        z = a
    else:
        n = a.size
        z = None
        if width_hint and n % width_hint == 0:
            z = a.reshape(n // width_hint, width_hint)
        elif length_hint and n % length_hint == 0:
            z = a.reshape(length_hint, n // length_hint)
        else:
            # Fallback: prefer 3200 as width if divisible; else try 1600; else square-ish
            for w_try in (3200, 1600):
                if n % w_try == 0:
                    z = a.reshape(n // w_try, w_try)
                    break
            if z is None:
                # last resort: keep as 1D and fail clearly
                raise ValueError(f"Can't infer 2D shape from {n} elements (no suitable width/length).")
    # Normalize orientation -> (length, width) with width as the larger dim (typical 3200)
    L, W = z.shape
    if L > W:  # looks transposed; want W >= L
        z = z.T
        L, W = z.shape
    return z  # shape (length, width)

def main():
    z_data_path_calibration = "/home/RUS_CIP/st184634/software_projects/laser_scanners/pc_calibration/z_data.npy"
    outlier_remover = OutlierRemover(remove_invalid=remove_invalid)
    # Load calibration FIRST and use its shape as ground truth
    z_calib = load_z_2d(z_data_path_calibration)
    length, width = z_calib.shape
    mask_calib = outlier_remover.remove_outliers(z_calib) # mask for calibration data with outliers removed
    # Find the workpiece and compute XY scale (mm/px) 
    scale_callibrator = Callibrator()
    bbox = scale_callibrator.detect_bbox(z_calib, mask_calib) # use calibration data to find the workpiece
    sx, sy = scale_callibrator.compute_scales_from_bbox(bbox) # compute scales from the detected bbox
    sy *= 0.8
    print(f"sx={sx:.6f} mm/px, sy={sy:.6f} mm/px")
    #sx, sy = 0.07, 0.14 # Override with known scales (mm/px)
    if visualise:
        scale_callibrator.show_bbox(z_calib, mask_calib, p_lo=20, p_hi=99, cmap="viridis") # visualize the bbox on calibration data
    counter = 0
    folders = []
    for base_path in glob("/mnt/nas/uncompressed_data/*/Keyence-Messung/"):
        folders.extend(sorted(glob(base_path + "*/")))
    for f in tqdm(folders):
        # The Operation 10 and the corresponding Operation 20 are not in the exact same directory, therefore this mappong occurs
        pairs = quick_get_pairs(f)
        for f_10, f_20 in pairs:
            # Format: 4 digits with leading zeros
            idx_str = f"{counter:04d}"
            run_my_logic("OP_10", outlier_remover, length, width, sx, sy, bbox=bbox, z_data_path=f"{f_10}/z_data.npy", idx_str=idx_str)
            run_my_logic("OP_20", outlier_remover, length, width, sx, sy, bbox=bbox, z_data_path=f"{f_20}/z_data.npy", idx_str=idx_str)
            counter += 1
            #break
        #break
    #run_my_logic("OP_10", outlier_remover, length, width, sx, sy, bbox=bbox, z_data_path="/home/RUS_CIP/st184634/software_projects/laser_scanners/pc_measurement/z_data.npy")
if __name__ == "__main__":
    main()

