import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage as ndi

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

        z = np.asarray(z, dtype=float)
        valid = np.isfinite(z) # Start with all finite points

        # Treat global min as invalid. Remove all points that are the lowest (= invalid values)
        zmin = np.nanmin(z)
        if self.remove_invalid:
            valid &= (z != zmin) # updates valid by AND-ing it with the new mask.

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

