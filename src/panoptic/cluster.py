"""Offset-shift clustering: thing points are shifted by the predicted offset toward their
instance center, then grouped with DBSCAN. Stuff points keep semantic-only labels.

This is the simple, stable baseline (DESIGN §2). A learned dynamic-shift (DS-Net) can replace
DBSCAN later as an ablation.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from ..data.semantic_kitti import THING_TRAIN_IDS


def panoptic_from_offsets(
    xyz: np.ndarray,          # [N,3]
    sem_pred: np.ndarray,     # [N] train ids 0..19
    offset_pred: np.ndarray,  # [N,3] offset toward instance center
    eps: float = 0.6,
    min_points: int = 20,
) -> np.ndarray:
    """Return instance ids [N] (0 = stuff/no-instance; >0 = instance id, unique per scan)."""
    inst = np.zeros(len(xyz), dtype=np.int64)
    shifted = xyz + offset_pred
    next_id = 1
    for cls in THING_TRAIN_IDS:
        idx = np.where(sem_pred == cls)[0]
        if len(idx) < min_points:
            continue
        labels = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1).fit_predict(shifted[idx])
        for c in np.unique(labels):
            if c == -1:  # DBSCAN noise
                continue
            inst[idx[labels == c]] = next_id
            next_id += 1
    return inst
