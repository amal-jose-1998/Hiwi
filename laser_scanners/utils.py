import os, sys
import numpy as np
import matplotlib.lines as mlines

def _in_ipykernel():
    try:
        import ipykernel 
        return True
    except Exception:
        return False

def _is_ssh():
    return any(k in os.environ for k in ("SSH_CONNECTION", "SSH_TTY", "SSH_CLIENT"))

def _setup_backend():
    """
    Behavior:
      - SSH + Jupyter/Interactive: ipympl (interactive inline)
      - Jupyter (local) without SSH: inline (static), don't force ipympl
      - Script with display: GUI (normal plt.show())
      - Script without display: Agg (headless)
    """
    import matplotlib
    os.environ.pop("MPLBACKEND", None)  # avoid forced Agg

    if _in_ipykernel():
        if _is_ssh():
            # Remote via SSH + Interactive Window -> go fully interactive
            try:
                import ipympl  # noqa: F401
                matplotlib.use("module://ipympl.backend_nbagg")
                return "interactive"
            except Exception:
                # Fall back to inline if ipympl not installed in remote kernel
                matplotlib.use("module://matplotlib_inline.backend_inline")
                return "inline"
        else:
            # Local Interactive Window: keep it simple/static
            matplotlib.use("module://matplotlib_inline.backend_inline")
            return "inline"

    # Not in Jupyter -> script mode
    if os.environ.get("DISPLAY") or sys.platform.startswith(("win", "darwin")):
        return "gui"  # normal GUI backend selected by Matplotlib
    matplotlib.use("Agg")
    return "headless"

_BACKEND_MODE = _setup_backend()

import matplotlib
print(f"[plot mode] {_BACKEND_MODE} | backend={matplotlib.get_backend()}")
import matplotlib.pyplot as plt

def maybe_show(fig):
        if _BACKEND_MODE in ("gui", "interactive", "inline"):
            try:
                fig.canvas.toolbar_visible = True
                fig.canvas.header_visible = False
            except Exception:
                pass
            plt.show()
        else:
            print("(Headless SSH) — skipping plot display.")

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
    blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',markersize=5, label='Valid (Kept) Points')
    red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',markersize=5, label='Removed Points')
    ax.legend(handles=[blue_dot, red_dot], loc='upper right', markerscale=3, fontsize=8)
    ax.set_title(plot_heading, fontsize=9, wrap=True)
    maybe_show(fig)

def plot_original_point_cloud(length, width, z_data_path, resolution , title, remove_invalid=True):
    z_vector = np.load(z_data_path)
    n = z_vector.size
    if n == length * width:
        h, w = length, width
    elif n == width * length:
        h, w = width, length
    else:
        if n % 3200 == 0:
            w = 3200
            h = n // w
        elif n % 1600 == 0:
            w = 1600
            h = n // w
        else:
            raise ValueError(f"Cannot infer 2D shape from {n} elements. "f"Given length={length}, width={width}")

    # Reshape and build grid
    z = z_vector.reshape(h, w)
    y, x = np.indices(z.shape)
    X, Y, Z = x.ravel(), y.ravel(), z.ravel()
    points = np.column_stack((X, Y, Z))

    # Remove all points that are the lowest (= invalid values)
    if remove_invalid:
        points = points[points[:,2]> points[:,2].min()]

    sub_sample_mask = np.random.choice(range(points.shape[0]), resolution)
    points = points[sub_sample_mask]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x, y, z = points[:,0], points[:,1], points[:,2]
    ax.scatter(x, y, z, cmap='viridis', c=z, s=1)
    ax.set_title(f"Original Point Cloud (Unfiltered) from {title}", fontsize=9, wrap=True)
    ax.set_xlabel("X [px]")
    ax.set_ylabel("Y [px]")
    ax.set_zlabel("Z [raw units]")
    maybe_show(fig)