"""SemanticKITTI panoptic dataset.

Returns per-scan points + panoptic targets. Voxelization / sparse-tensor collation is left to the
model/backbone stage (torchsparse) so this loader stays framework-agnostic and unit-testable.

Layout expected:
    {root}/sequences/{seq}/velodyne/{frame}.bin   float32 [N,4] (x,y,z,remission)
    {root}/sequences/{seq}/labels/{frame}.label   uint32  [N]   (train split only)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .semantic_kitti import is_thing, remap_semantic, split_label

# standard SemanticKITTI splits
SPLITS: dict[str, list[str]] = {
    "train": [f"{i:02d}" for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]],
    "val": ["08"],
    "test": [f"{i:02d}" for i in range(11, 22)],
}


class SemanticKITTIPanoptic(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root)
        self.split = split
        self.frames: list[tuple[str, str]] = []
        for seq in SPLITS[split]:
            vel_dir = self.root / "sequences" / seq / "velodyne"
            if not vel_dir.is_dir():
                continue
            for bin_path in sorted(vel_dir.glob("*.bin")):
                self.frames.append((seq, bin_path.stem))
        if not self.frames:
            raise FileNotFoundError(f"no scans under {self.root}/sequences for split={split}")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> dict:
        seq, frame = self.frames[i]
        pts = np.fromfile(
            self.root / "sequences" / seq / "velodyne" / f"{frame}.bin", dtype=np.float32
        ).reshape(-1, 4)
        xyz, remission = pts[:, :3], pts[:, 3:4]

        sample = {"seq": seq, "frame": frame, "xyz": xyz, "feat": remission}

        label_path = self.root / "sequences" / seq / "labels" / f"{frame}.label"
        if label_path.exists():
            raw = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
            raw_sem, inst = split_label(raw)
            sem = remap_semantic(raw_sem)  # 0..19
            # instance id 0 for stuff/ignore; keep original per-thing instance id otherwise
            inst = np.where(is_thing(sem), inst, 0).astype(np.int64)
            sample.update({"sem": sem, "inst": inst})
        return sample
