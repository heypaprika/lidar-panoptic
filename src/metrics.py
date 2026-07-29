"""Semantic mIoU meter (confusion-matrix based), ignoring the ignore class (id 0)."""

from __future__ import annotations

import numpy as np


class IoUMeter:
    def __init__(self, num_classes: int, ignore: int = 0):
        self.n = num_classes
        self.ignore = ignore
        self.conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        valid = gt != self.ignore
        p, g = pred[valid].astype(np.int64), gt[valid].astype(np.int64)
        k = g * self.n + p
        self.conf += np.bincount(k, minlength=self.n * self.n).reshape(self.n, self.n)

    def compute(self) -> tuple[np.ndarray, float]:
        tp = np.diag(self.conf).astype(np.float64)
        fp = self.conf.sum(0) - tp
        fn = self.conf.sum(1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1.0)
        classes = [c for c in range(self.n) if c != self.ignore]
        return iou, float(np.mean(iou[classes]))

    def reset(self) -> None:
        self.conf[:] = 0
