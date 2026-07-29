"""Panoptic Quality (PQ) evaluation — wrap the official SemanticKITTI evaluator.

Do NOT hand-roll PQ: void handling and the >0.5-IoU matching are subtle (DESIGN §4). Vendor or
pip-install PRBonn/semantic-kitti-api and call its PanopticEval; this module just adapts our
(sem, inst) arrays to it and accumulates per scan.

Reference: PRBonn/semantic-kitti-api → auxiliary/np_ioueval.py (PanopticEval), evaluate_panoptic.py
"""

from __future__ import annotations

import numpy as np

from ..data.semantic_kitti import IGNORE_ID, NUM_CLASSES, THING_TRAIN_IDS


class PanopticScorer:
    """Thin adapter. Accumulate (pred, gt) per scan, then .summary() -> PQ/SQ/RQ/mIoU."""

    def __init__(self):
        # TODO(gate2): from semantic_kitti_api.auxiliary.np_ioueval import PanopticEval
        #   self.evaluator = PanopticEval(NUM_CLASSES, ignore=[IGNORE_ID], min_points=...)
        self.ignore = [IGNORE_ID]
        self.things = sorted(THING_TRAIN_IDS)
        self._n = NUM_CLASSES
        raise NotImplementedError("Wrap PRBonn semantic-kitti-api PanopticEval. See DESIGN.md §4.")

    def add(self, sem_pred: np.ndarray, inst_pred: np.ndarray,
            sem_gt: np.ndarray, inst_gt: np.ndarray) -> None:
        # self.evaluator.addBatch(sem_pred, inst_pred, sem_gt, inst_gt)
        raise NotImplementedError

    def summary(self) -> dict[str, float]:
        # pq, sq, rq, all_pq(per-class), miou = self.evaluator.getPQ(); ... = self.evaluator.getSemIoU()
        raise NotImplementedError
