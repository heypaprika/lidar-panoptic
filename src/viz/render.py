"""Open3D visualization: color points by semantic class or by instance id.

Usage (after eval produces preds): render semantic vs panoptic side by side for demo frames.
"""

from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
except ImportError:  # keep import-safe on headless CI
    o3d = None


def _palette(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n + 1, 3))


def _bev_fig(xyz, colors, order, path, title, point_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xyz[order, 0], xyz[order, 1], s=point_size, c=colors[order], linewidths=0)
    ax.set_aspect("equal")
    ax.set(title=title, xlabel="x (m)", ylabel="y (m)")
    ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_bev(xyz: np.ndarray, labels: np.ndarray, path: str, seed: int = 0,
             title: str = "", point_size: float = 0.6) -> None:
    """Headless bird's-eye-view (top-down x-y) scatter via matplotlib — no EGL/GPU needed.
    Fallback for boxes without libEGL (Open3D OffscreenRenderer). labels = semantic or instance ids."""
    colors = _palette(int(labels.max()), seed)[labels]
    _bev_fig(xyz, colors, np.argsort(labels), path, title, point_size)  # background (id 0) first


def save_bev_panoptic(xyz: np.ndarray, sem: np.ndarray, inst: np.ndarray, path: str,
                      title: str = "", point_size: float = 0.6) -> None:
    """Proper panoptic BEV: stuff points colored by semantic class, thing instances each a distinct
    color. Headless (matplotlib). This is the panoptic convention — not a single-color background."""
    sem_pal = _palette(int(sem.max()), seed=1)
    inst_pal = _palette(int(inst.max()), seed=7)
    colors = sem_pal[sem].copy()               # stuff & background: semantic color
    thing = inst > 0
    colors[thing] = inst_pal[inst[thing]]      # thing instances: per-instance color
    order = np.argsort(thing.astype(int))      # draw stuff first, instances on top
    _bev_fig(xyz, colors, order, path, title, point_size)


def _cloud(xyz: np.ndarray, labels: np.ndarray, seed: int):
    colors = _palette(int(labels.max()), seed)[labels]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc


def show(xyz: np.ndarray, labels: np.ndarray, seed: int = 0) -> None:
    """labels = semantic ids or instance ids; colored by a fixed palette (interactive window)."""
    if o3d is None:
        raise ImportError("pip install open3d")
    o3d.visualization.draw_geometries([_cloud(xyz, labels, seed)])


def save(xyz: np.ndarray, labels: np.ndarray, path: str, seed: int = 0,
         size: tuple[int, int] = (1280, 960)) -> None:
    """Offscreen render to a PNG — for headless cloud boxes (needs EGL/GPU)."""
    if o3d is None:
        raise ImportError("pip install open3d")
    pc = _cloud(xyz, labels, seed)
    rnd = o3d.visualization.rendering.OffscreenRenderer(*size)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader, mat.point_size = "defaultUnlit", 2.0
    rnd.scene.set_background([1.0, 1.0, 1.0, 1.0])
    rnd.scene.add_geometry("pc", pc, mat)
    bounds = pc.get_axis_aligned_bounding_box()
    rnd.setup_camera(60.0, bounds, bounds.get_center())
    o3d.io.write_image(path, rnd.render_to_image())
