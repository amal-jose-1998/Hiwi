import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from outlier_correction import OutlierRemover
from scale_correction import Callibrator
from reflection_remover import remove_reflections, visualise_removed_points
from glob import glob
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
import utils

base_path = r"/mnt/nas/uncompressed_data/real_01/Keyence-Messung/"

# Laser Sensor: 
# - 3200 points per line
# - 1600 lines in total
width = 3200
length = 1600

resolution = 20000 # Number of points to random sample. Increased=more details, Decreased=faster visualization.
remove_invalid = True # If True remove points, that are invalid (like double.nan)
scaling_factor_in_Z = 30 # Scale Z to this value (mm)

def run_my_logic(outlier_remover, length, width, sx, sy, bbox = None, z_data_path = "/home/RUS_CIP/st184634/software_projects/laser_scanners/pc_measurement/z_data.npy"):
    #utils.plot_original_point_cloud(length, width, z_data_path, resolution)
    pcd = o3d.geometry.PointCloud()
    z = load_z_2d(z_data_path, width_hint=width, length_hint=length) 
    y, x = np.indices(z.shape) 
    # Prepare XYZ from kept pixels
    X_px_all = x.ravel()
    Y_px_all = y.ravel()
    Z_all = z.ravel()
    # Remove outliers
    mask = outlier_remover.remove_outliers(z) # mask for real measurement with outliers removed
    outlier_removed = ~mask.ravel()
    
    X_px, Y_px, Z = x[mask], y[mask], z[mask]

    # Apply scaling
    X = X_px * sx
    Y = Y_px * sy

    z_min = float(Z.min())
    z_max = float(Z.max())
    den   = max(z_max - z_min, 1e-9)
    Z  = (Z - z_min) / den * scaling_factor_in_Z
    points = np.column_stack((X, Y, Z))
    
    original_points = np.column_stack((X_px_all * sx, Y_px_all * sy, ((Z_all - z_min) / den) * scaling_factor_in_Z))
    visualise_removed_points(original_points, outlier_removed, resolution, plot_heading="red = outliers")

    #z = points[:, -1]
    #z -= np.min(z) # remove offset
    #z = (z / np.max(z)) * scaling_factor_in_Z 
    #points[:, -1] = z

    #Remove reflection artifacts from the point cloud
    refined_points, valid_mask = remove_reflections(points, band_mm=30.0)
    reflection_removed = ~valid_mask # 1D mask of reflection-removed points
    
    reflection_points = points[reflection_removed]
    outlier_points = np.column_stack((X_px_all[outlier_removed] * sx, Y_px_all[outlier_removed] * sy,((Z_all[outlier_removed] - z_min) / den) * scaling_factor_in_Z))    
    all_removed_points = np.vstack((outlier_points, reflection_points))

    # Subsample for visualization
    if all_removed_points.shape[0] > resolution:
        idx = np.random.choice(all_removed_points.shape[0], resolution, replace=False)
        all_removed_points = all_removed_points[idx]
    if refined_points.shape[0] > resolution:
        idx = np.random.choice(refined_points.shape[0], resolution, replace=False)
        refined_points = refined_points[idx]
    #utils.visualise_removed_points(all_removed_points, refined_points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(refined_points[:,0], refined_points[:,1], refined_points[:,2], c=refined_points[:,2], cmap='viridis', s=1)

    ax.set_xlabel("X [mm]" if bbox is not None else "X [px]")
    ax.set_ylabel("Y [mm]" if bbox is not None else "Y [px]")
    ax.set_zlabel("Z [mm]")
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
    #scale_callibrator.show_bbox(z_calib, mask_calib, p_lo=20, p_hi=99, cmap="viridis") # visualize the bbox on calibration data

    folders = [f for f in sorted(glob(base_path + "*/"))]
    for f in tqdm(folders):
        # The Operation 10 and the corresponding Operation 20 are not in the exact same directory, therefore this mappong occurs
        pairs = quick_get_pairs(f)
        for f_10, f_20 in pairs:
            run_my_logic(outlier_remover, length, width, sx, sy, z_data_path=f"{f_10}/z_data.npy")
            #run_my_logic(outlier_remover, length, width, sx, sy, z_data_path=f"{f_20}/z_data.npy")
            break
        break
    #run_my_logic(outlier_remover, length, width, sx, sy, z_data_path="/home/RUS_CIP/st184634/software_projects/laser_scanners/pc_measurement/z_data.npy")

if __name__ == "__main__":
    main()

