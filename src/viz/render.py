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


def show(xyz: np.ndarray, labels: np.ndarray, seed: int = 0) -> None:
    """labels = semantic ids or instance ids; colored by a fixed palette."""
    if o3d is None:
        raise ImportError("pip install open3d")
    colors = _palette(int(labels.max()), seed)[labels]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pc])
