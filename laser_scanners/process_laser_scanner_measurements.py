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
import open3d as o3d

base_path = r"/mnt/nas/uncompressed_data/real_01/Keyence-Messung/"

# Laser Sensor: 
# - 3200 points per line
# - 1600 lines in total
width = 3200
length = 1600

resolution = 20000 # Number of points to random sample. Increased=more details, Decreased=faster visualization.
remove_invalid = True # If True remove points, that are invalid (like double.nan)
scaling_factor_in_Z = 30 # Scale Z to this value (mm)
visualise = True # If True, visualise the detected bounding box, the workpiece, removed outliers from the point cloud and the final point cloud

def apply_scaling(X_px, Y_px, Z, sx, sy, scaling_factor_in_Z):
    """Apply scaling to pixel coordinates and Z values."""
    X = X_px * sx
    Y = Y_px * sy

    z_min = float(Z.min())
    z_max = float(Z.max())
    den   = max(z_max - z_min, 1e-9)
    Z_scaled  = (Z - z_min) / den * scaling_factor_in_Z
    return np.column_stack((X, Y, Z_scaled))

def run_my_logic(op_tag, outlier_remover, length, width, sx, sy, bbox = None, z_data_path = "/home/RUS_CIP/st184634/software_projects/laser_scanners/pc_measurement/z_data.npy"):
    if visualise:
        utils.plot_original_point_cloud(length, width, z_data_path, resolution, title = z_data_path)
    z = load_z_2d(z_data_path, width_hint=width, length_hint=length) 
    y, x = np.indices(z.shape) 
    # Prepare XYZ from kept pixels
    X_px_all = x.ravel()
    Y_px_all = y.ravel()
    Z_all = z.ravel()
    # Remove outliers
    if op_tag == "OP_10":
        mask = outlier_remover.remove_outliers(z) # mask for real measurement with outliers removed
        band_mm = 25.0
    else:
        mask = outlier_remover.remove_invalid_z(z) # mask for real measurement with outliers removed
        band_mm = 17.0
    outlier_removed = ~mask.ravel()
    X_px, Y_px, Z = x[mask], y[mask], z[mask]

    # Apply scaling
    points = apply_scaling(X_px, Y_px, Z, sx, sy, scaling_factor_in_Z) #scaled points after outlier removal
    original_points = apply_scaling(X_px_all, Y_px_all, Z_all, sx, sy, scaling_factor_in_Z) #scaled original points before outlier removal
    
    if visualise:
        utils.visualise_removed_points(original_points, outlier_removed, resolution, plot_heading=f"Outliers removed from {z_data_path}")

    #Remove reflection artifacts from the point cloud
    refined_points = remove_reflections(op_flag=op_tag, points=points, z_data_path=z_data_path, band_mm=band_mm, visualise=visualise)
    
    if refined_points is None or refined_points.size == 0:
        raise ValueError(f"No points left after reflection removal for {z_data_path}.")

    op_folder = Path(z_data_path).parent.name   
    parent_folder = Path(z_data_path).parent.parent.name
    scan_id = f"{parent_folder}_{op_folder}"
    save_dir = Path("/home/RUS_CIP/st184634/software_projects/laser_scanners/refined_data/")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"refined_points_{scan_id}.npy"
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

    folders = [f for f in sorted(glob(base_path + "*/"))]
    for f in tqdm(folders):
        # The Operation 10 and the corresponding Operation 20 are not in the exact same directory, therefore this mappong occurs
        pairs = quick_get_pairs(f)
        for f_10, f_20 in pairs:
            run_my_logic("OP_10", outlier_remover, length, width, sx, sy, z_data_path=f"{f_10}/z_data.npy")
            run_my_logic("OP_20", outlier_remover, length, width, sx, sy, z_data_path=f"{f_20}/z_data.npy")
            break
        #break
    #run_my_logic("OP_10", outlier_remover, length, width, sx, sy, z_data_path="/home/RUS_CIP/st184634/software_projects/laser_scanners/pc_measurement/z_data.npy")
if __name__ == "__main__":
    main()

