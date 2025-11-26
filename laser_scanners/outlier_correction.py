import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage as ndi
import open3d as o3d
import utils
from scipy.spatial import cKDTree

class OutlierRemover:
    """
    Outlier removal using connected components + DBSCAN.

    Parameters
    min_samples : int, default=1
        Minimum samples parameter for DBSCAN. With 1, even isolated components can form clusters (helps retain small but valid structures).
    eps : float or None, default=None
        Neighborhood radius for DBSCAN. If None, it is estimated automatically from centroids using `_auto_eps_centroids`.
    k_neighbors : int, default=4
        Number of nearest neighbors used to estimate `eps`.
    eps_clip : tuple of (float, float), default=(2.0, 200.0)
        Minimum and maximum values for clipping the automatically estimated `eps`.
    """

    def __init__(self, min_samples=1, eps=None, k_neighbors=4, eps_clip=(2.0, 200.0), remove_invalid=True):
        self.min_samples = min_samples
        self.eps = eps
        self.k_neighbors = k_neighbors
        self.eps_clip = eps_clip
        self.remove_invalid = remove_invalid
    
    def _auto_eps_centroids(self, centroids_xy):
        """
        Helper to pick a DBSCAN eps automatically from the centroid coordinates.

        Parameters
        centroids_xy : ndarray of shape (n_components, 2)
            Array of (x, y) coordinates of connected-component centroids.
        
        Returns
        eps : float
            Estimated neighborhood radius for DBSCAN, clipped to `eps_clip`.
        """

        if centroids_xy.shape[0] <= 1: # k-NN is meaningless with 0 or 1 point
            return 10.0 # arbitrary value
        k = min(self.k_neighbors, max(1, centroids_xy.shape[0])) # to make sure that k dont exceed the number of points.
        nn = NearestNeighbors(n_neighbors=k).fit(centroids_xy)
        dists, _ = nn.kneighbors(centroids_xy) # distances to k nearest neighbors
        kth = dists[:, -1] # k-th nearest neighbor distance
        eps = float(np.median(kth) * 1.5) # 1.5 is a heuristic factor for robustness
        return float(np.clip(eps, 2.0, 200.0)) # clamp to reasonable range

    def remove_invalid_z(self, z):
        """
        Remove invalid points from z (like global min or NaN).

        Parameters
        z : ndarray of shape (H, W)
            2D array of depth values (floats).

        Returns
        valid_mask : ndarray of shape (H, W), dtype=bool
            Boolean mask where True marks valid (finite and non-min) pixels.
        """
        z = np.asarray(z, dtype=float)
        valid = np.isfinite(z) # Start with all finite points

        # Treat global min as invalid. Remove all points that are the lowest (= invalid values)
        zmin = np.nanmin(z)
        if self.remove_invalid:
            valid &= (z != zmin) # updates valid by AND-ing it with the new mask.

        return valid

    def remove_outliers(self, z):
        """
        Remove outliers using connected components and DBSCAN on centroids.

        Parameters
        z : ndarray of shape (H, W)
            2D array of depth values (floats).
        min_samples : int, optional, default=1
            Minimum samples parameter for DBSCAN. With 1, even isolated components can form clusters (helps retain small but valid structures).
        eps : float or None, optional
            Neighborhood radius for DBSCAN. If None, it is automatically estimated from component centroids using `_auto_eps_centroids`.

        Returns
        keep_mask : ndarray of shape (H, W), dtype=bool
            Boolean mask where True marks pixels belonging to the retained components.
        """

        valid = self.remove_invalid_z(z) # initial valid mask

        labeled, ncomp = ndi.label(valid) # connected component labeling
        
        if ncomp == 0:   
            return valid # if no blobs are found, just return the original valid mask.

        sizes = np.bincount(labeled.ravel()) # count how many pixels each component has
        sizes[0] = 0 # background component is labeled 0, ignore it

        ys, xs = np.nonzero(labeled) # pixel coordinates of valid points.
        labs = labeled[ys, xs] # component labels of valid points.
        max_lab = int(labeled.max()) # maximum component label
    
        # Compute centroids of all components
        sum_x = np.bincount(labs, weights=xs, minlength=max_lab + 1)
        sum_y = np.bincount(labs, weights=ys, minlength=max_lab + 1) 
        cnt   = np.bincount(labs, minlength=max_lab + 1) # count of pixels per component
        cnt[0] = 0 # ignore background
        valid_labels = np.where(cnt > 0)[0]  # valid centroid labels

        if valid_labels.size == 0: 
            return valid # guard against empty centroids

        centroids = np.column_stack((sum_x[valid_labels] / cnt[valid_labels], sum_y[valid_labels] / cnt[valid_labels]))
        comp_sizes = sizes[valid_labels]# size for each centroid

        # DBSCAN on centroids
        eps_used = self.eps if self.eps is not None else self._auto_eps_centroids(centroids)
        db = DBSCAN(eps=eps_used, min_samples=self.min_samples) # DBSCAN model on centroids.
        c_labels = db.fit_predict(centroids)  # one label per component

        # If DBSCAN found no clusters (all noise), just keep all components
        if not np.any(c_labels >= 0):
            return (labeled > 0)

        # Sum sizes within each centroid-cluster; pick the majority
        uniq = np.unique(c_labels[c_labels >= 0]) # unique cluster IDs, excluding noise (-1)
        totals = [comp_sizes[c_labels == cid].sum() for cid in uniq] # total size per cluster
        majority_cid = uniq[int(np.argmax(totals))] # cluster ID of the largest cluster
        kept_component_labels = valid_labels[c_labels == majority_cid] # component labels to keep

        keep_mask = np.isin(labeled, kept_component_labels) # final mask of pixels to keep
        return keep_mask

    def gradient_z_filter(self, z, threshold=8.0):
        """
        Remove pixels where Z changes too abruptly.
        Returns a boolean mask: True = keep, False = remove.

        Parameters
        z : 2D numpy array
            Depth map (length * width)
        threshold : float
            Gradient magnitude threshold. 
            Typical values:
                5-8  : mild removal
                10-15: keep more edges
                20+  : only remove extreme reflections
        """
        # Compute gradients along X and Y directions
        gx, gy = np.gradient(z)
        # Gradient magnitude
        mag = np.sqrt(gx**2 + gy**2)
        # Keep pixels with small/smooth gradients
        mask = mag < threshold
        return mask
    
    def o3d_statistical_cleanup(self, points, z_data_path=None, visualise=True, resolution=20000, nb_neighbors=20, std_ratio=3 ):
        """
        Apply Open3D statistical outlier removal on a point cloud.

        points       : (N, 3) numpy array
        z_data_path  : used only for plot title / debugging
        nb_neighbors : number of neighbors to analyze for each point
        std_ratio    : standard deviation ratio threshold
        """
        # Build Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # Statistical outlier removal
        clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        clean_points = np.asarray(clean_pcd.points)

        if visualise:
            keep_mask = np.zeros(len(points), dtype=bool)
            keep_mask[ind] = True
            remove_mask = ~keep_mask
            title = (
                f"Statistical outliers removed from {z_data_path}"
                if z_data_path is not None
                else "Statistical outliers removed"
            )
            utils.visualise_removed_points(points, remove_mask, resolution, plot_heading=title)

        return clean_points  
    
    def local_planarity_filter(self, points, z_min=0.0, z_max=10, k=15, curvature_threshold=0.015, visualise=True, resolution=20000, title="Local planarity filter"):
        """
        Removes points whose local neighborhood is not planar.
        Uses PCA on each local patch and removes points with high curvature.
        
        curvature_threshold:
            Lower → more aggressive removal.
            Typical values: 0.01-0.04
        """

        pts = np.asarray(points)

        # restrict to leg band
        band_mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
        if not np.any(band_mask):
            return pts

        band_points = pts[band_mask]
        other_points = pts[~band_mask]

        M = len(band_points)
        if M <= k + 1:
            return pts  # too few to do anything 
    
        tree = cKDTree(band_points)
        keep_band = np.ones(M, dtype=bool)

        for i in range(M):
            _, nbr_idx = tree.query(band_points[i], k=k+1)
            nbrs = band_points[nbr_idx]

            C = np.cov(nbrs.T)
            eigvals = np.linalg.eigvalsh(C)
            curvature = eigvals[0] / (eigvals.sum() + 1e-9)

            if curvature > curvature_threshold:
                keep_band[i] = False

        band_filtered = band_points[keep_band]
        if visualise:
            removed_indices = np.nonzero(~keep_band)[0]
            global_remove_mask = np.zeros(len(pts), dtype=bool)
            # map back to actual indices in input `points`
            band_global_indices = np.nonzero(band_mask)[0]
            global_remove_mask[ band_global_indices[removed_indices] ] = True
            utils.visualise_removed_points(pts, global_remove_mask, resolution, plot_heading=title, )

        return np.vstack([other_points, band_filtered])