import numpy as np
import utils
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi

class Callibrator:
    """
    Detect the highest-Z blob (via top percentile), compute XY scales from a known 30*9 mm insert, and visualize the bbox with percentile-based coloring.
    
    Parameters  
    long_mm : float, default=30.0
        Length of the longer side of the workpiece in mm.
    short_mm : float, default=9.0
        Length of the shorter side of the workpiece in mm.
    top_q : float, default=99.0 
        Percentile for detecting the top pixels of the workpiece.
    """

    def __init__(self, long_mm=30.0, short_mm=9.0, top_q=99.0):
        self.long_mm = float(long_mm)
        self.short_mm = float(short_mm)
        self.top_q = float(top_q)
        self.bbox = None          # (xmin, xmax, ymin, ymax)
        self.sx = 1.0             # mm/px (X)
        self.sy = 1.0             # mm/px (Y)

    def detect_bbox(self, z, valid_mask):
        """
        Detect the bbox of the highest-Z blob in `z` using `valid_mask`.

        Parameters
        z : ndarray of shape (H, W)
            2D array of depth values (floats).
        valid_mask : ndarray of shape (H, W), dtype=bool
            Boolean mask where True marks valid pixels.

        Returns
        bbox : tuple of (xmin, xmax, ymin, ymax) or None
            Bounding box of the detected blob in pixel coordinates, or None if detection failed.
        """

        if not np.any(valid_mask):
            self.bbox = None
            return None
        # Keep only pixels at the very top in Z.
        thr = np.percentile(z[valid_mask], self.top_q) # threshold for top q-percentile
        top = valid_mask & (z >= thr)

        lbl, n = ndi.label(top) # connected component labeling
        if n == 0:
            self.bbox = None
            return None

        best = None # (bbox, meanZ)
        for k in range(1, n+1):
            ys, xs = np.nonzero(lbl == k)
            # bbox in pixel coordinates
            xmin, xmax = xs.min(), xs.max() 
            ymin, ymax = ys.min(), ys.max() 
            meanZ = float(z[ys, xs].mean()) # mean Z of this component
            cand = ((xmin, xmax, ymin, ymax), meanZ) # candidate bbox
            # Pick the best candidate: highest meanZ
            if (best is None) or (cand[1] > best[1]): 
                best = cand 

        self.bbox = None if best is None else best[0]
        return self.bbox
    
    def compute_scales_from_bbox(self, bbox):
        """
        Compute XY scales (mm/px) from the detected bbox.

        Parameters
        bbox : tuple of (xmin, xmax, ymin, ymax) or None
            Bounding box of the detected blob in pixel coordinates.
        
        Returns
        sx, sy : float, float
            Scale factors in mm/px for X and Y axes. If bbox is None, returns (1.0, 1.0).
        """

        if bbox is None:
            bbox = self.bbox
        if bbox is None:
            self.sx, self.sy = 1.0, 1.0
            return self.sx, self.sy
        
        xmin, xmax, ymin, ymax = bbox
        w_px = (xmax - xmin + 1)
        h_px = (ymax - ymin + 1)

        if w_px >= h_px:
            self.sx = self.long_mm / w_px   # x spans the 30 mm side
            self.sy = self.short_mm / h_px  # y spans the 9 mm side
        else:
            self.sx = self.short_mm / w_px  # x spans the 9 mm side
            self.sy = self.long_mm / h_px   # y spans the 30 mm side
        return self.sx, self.sy
    
    def show_bbox(self, z, valid_mask, p_lo=0.5, p_hi=99.5, cmap="viridis"):
        """
        Visualize the detected bbox on top of the Z image, with percentile-based color scaling.
        
        Parameters
        z : ndarray of shape (H, W)         
            2D array of depth values (floats).
        valid_mask : ndarray of shape (H, W), dtype=bool    
            Boolean mask where True marks valid pixels.
        p_lo, p_hi : float, optional, default=0.5, 99.5
            Percentiles for color scaling.
        cmap : str or Colormap, optional, default="viridis
            Colormap for visualization.
        """
       
        bbox = self.bbox

        vmin = float(np.percentile(z[valid_mask], p_lo))
        vmax = float(np.percentile(z[valid_mask], p_hi))
        # if percentiles are invalid, revert to min/max of the good pixels.
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin: 
            vmin = float(np.nanmin(z[valid_mask])) 
            vmax = float(np.nanmax(z[valid_mask]))
            if vmax == vmin:
                vmax = vmin + 1e-6

        fig, ax = plt.subplots(figsize=(7.5, 6))
        im = ax.imshow(z, cmap=cmap, vmin=vmin, vmax=vmax)

        xmin, xmax, ymin, ymax = bbox # unpack bbox
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin + 1, ymax - ymin + 1, fill=False, ec='r', lw=2)) # draw a red rectangle on top of the image corresponding to the bbox

        ax.set_aspect('equal')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Z (sensor units)")
        plt.tight_layout()
        utils.maybe_show(fig)
