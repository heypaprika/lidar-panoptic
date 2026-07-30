"""SemanticKITTI label maps (panoptic).

Mirrors the official `semantic-kitti.yaml`. Raw labels are uint32 where the low 16 bits are the
semantic id and the high 16 bits are the instance id. `LEARNING_MAP` collapses raw semantic ids
(incl. the moving-* duplicates) into 20 training ids (0 = ignore/unlabeled, 1..19 = classes).
Instance ids only carry meaning for *thing* classes (train ids 1..8).
"""

from __future__ import annotations

import numpy as np

# raw semantic id -> training id (0 = ignore). Matches PRBonn semantic-kitti-api.
LEARNING_MAP: dict[int, int] = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7, 32: 8,
    40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17,
    80: 18, 81: 19, 99: 0,
    # moving-* variants map to their static class
    252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}

CLASS_NAMES: list[str] = [
    "ignore", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building",
    "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
]

NUM_CLASSES = 20            # incl. ignore at index 0
IGNORE_ID = 0
THING_TRAIN_IDS: set[int] = {1, 2, 3, 4, 5, 6, 7, 8}   # car..motorcyclist

# dense LUT for fast vectorized remap (raw id in [0, 259])
_LUT = np.zeros(260, dtype=np.int64)
for _raw, _train in LEARNING_MAP.items():
    _LUT[_raw] = _train


def remap_semantic(raw_sem: np.ndarray) -> np.ndarray:
    """raw semantic ids -> training ids (0..19)."""
    return _LUT[raw_sem]


def split_label(label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """uint32 .label -> (raw_semantic:int32, instance:int32)."""
    sem = (label & 0xFFFF).astype(np.int64)
    inst = (label >> 16).astype(np.int64)
    return sem, inst


def is_thing(train_ids: np.ndarray) -> np.ndarray:
    out = np.zeros_like(train_ids, dtype=bool)
    for t in THING_TRAIN_IDS:
        out |= train_ids == t
    return out
