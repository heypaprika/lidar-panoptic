"""Panoptic Quality (PQ) evaluation — thin adapter over the official SemanticKITTI evaluator.

Do NOT hand-roll PQ: void handling and the >0.5-IoU greedy matching are subtle (DESIGN §4). We
vendor PRBonn/semantic-kitti-api's `PanopticEval` (from auxiliary/eval_np.py — note: np_ioueval.py
only has the semantic `iouEval`) and only adapt our (sem, inst) arrays to it:

    wget -O src/panoptic/eval_np.py \\
      https://raw.githubusercontent.com/PRBonn/semantic-kitti-api/master/auxiliary/eval_np.py

`PanopticEval` works in *train-id* space (0..19, ignore=0) — the same space our predictions and
remapped GT already use — so no extra remap is needed here.
"""

from __future__ import annotations

import numpy as np

from ..data.semantic_kitti import IGNORE_ID, NUM_CLASSES, THING_TRAIN_IDS


def _load_eval_cls():
    try:
        from .eval_np import PanopticEval  # vendored from PRBonn/semantic-kitti-api auxiliary/eval_np.py
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Vendor the official evaluator first:\n"
            "  wget -O src/panoptic/eval_np.py "
            "https://raw.githubusercontent.com/PRBonn/semantic-kitti-api/master/auxiliary/eval_np.py"
        ) from e
    return PanopticEval


class PanopticScorer:
    """Accumulate (pred, gt) per scan, then .summary() -> PQ/PQ_dagger/SQ/RQ/mIoU (train-id space).

    Evaluator is created lazily on the first `.add()` so constructing a panoptic LightningModule
    (or importing this) doesn't require the vendored file until validation actually runs.
    """

    def __init__(self, min_points: int = 50):
        self.min_points = min_points
        self.things = sorted(THING_TRAIN_IDS)
        self._ev = None

    def _ensure(self):
        if self._ev is None:
            self._ev = _load_eval_cls()(NUM_CLASSES, ignore=[IGNORE_ID], min_points=self.min_points)

    def add(self, sem_pred: np.ndarray, inst_pred: np.ndarray,
            sem_gt: np.ndarray, inst_gt: np.ndarray) -> None:
        self._ensure()
        self._ev.addBatch(
            sem_pred.astype(np.int32), inst_pred.astype(np.int32),
            sem_gt.astype(np.int32), inst_gt.astype(np.int32),
        )

    def summary(self) -> dict[str, float]:
        self._ensure()
        pq, sq, rq, pq_cls, _, _ = self._ev.getPQ()          # scalars are class means
        miou, _ = self._ev.getSemIoU()
        # PQ† (dagger): stuff classes use IoU instead of PQ, averaged with thing PQ.
        thing = np.array(self.things)
        stuff = np.array([c for c in range(NUM_CLASSES) if c != IGNORE_ID and c not in self.things])
        iou_mean, iou_cls = self._ev.getSemIoU()
        pq_dagger = float((pq_cls[thing].sum() + iou_cls[stuff].sum()) / (len(thing) + len(stuff)))
        return {
            "PQ": float(pq), "PQ_dagger": pq_dagger,
            "SQ": float(sq), "RQ": float(rq), "mIoU": float(miou),
        }

    def per_class(self) -> dict[str, np.ndarray]:
        """Per-class vectors (index = train id 0..19): PQ, SQ, RQ, IoU."""
        self._ensure()
        _, _, _, pq_c, sq_c, rq_c = self._ev.getPQ()
        _, iou_c = self._ev.getSemIoU()
        return {"PQ": pq_c, "SQ": sq_c, "RQ": rq_c, "IoU": iou_c}

    def reset(self) -> None:
        self._ev = None
