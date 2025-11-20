from pathlib import Path
import numpy as np
import open3d as o3d


REFINED_DIR = Path("/home/RUS_CIP/st184634/software_projects/laser_scanners/refined_data")
MESH_DIR    = REFINED_DIR / "meshes"
MESH_DIR.mkdir(exist_ok=True)

def npy_to_pcd(npy_path):
    """Load a (N,3) numpy file and convert to Open3D point cloud."""
    pts = np.load(npy_path)          
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{npy_path} has shape {pts.shape}, expected (N, 3).")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def prepare_pcd(pcd, voxel_size=None):
    """Optional downsampling + normals for Poisson."""
    if voxel_size is not None:
        print("Points BEFORE downsampling:", len(pcd.points))
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print("Points AFTER downsampling:", len(pcd.points))

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(k=50)
    return pcd


def poisson_reconstruct(pcd, depth=9,trim_quantile=0.05):
    """Run Poisson and trim low-density vertices."""
    print("Running Poisson surface reconstruction...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    print(mesh)
    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, trim_quantile)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(mesh)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    return mesh, densities

def process_all_refined_clouds(voxel_size=0.3, depth=9, trim_quantile=0.05, visualize=True):
    npy_files = sorted(REFINED_DIR.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {REFINED_DIR}")

    print(f"Found {len(npy_files)} refined point clouds.")
    for f in npy_files:
        print(f"\n=== Processing {f.name} ===")
        pcd = npy_to_pcd(f)
        print(pcd)

        pcd = prepare_pcd(pcd, voxel_size=voxel_size)
        mesh, densities = poisson_reconstruct(pcd, depth=depth, trim_quantile=trim_quantile)
        # Save mesh as .ply next to the point cloud
        mesh_path = MESH_DIR / f"{f.stem}_mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_vertex_colors=True)
        print(f"Saved mesh to {mesh_path}")

        if visualize:
            print("Visualizing mesh...")
            o3d.visualization.draw([mesh])




if __name__ == "__main__":
    process_all_refined_clouds(
        voxel_size=None,     # tune based on your mm scale; None = no downsampling
        depth=9,            # lower if too slow, higher for more detail
        trim_quantile=0.05, # remove lowest 5% densest vertices
        visualize=False,
    )