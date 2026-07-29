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
