"""Write a tiny SemanticKITTI-format dataset to disk to shake out the *real* dataloader +
torchsparse backbone forward (backbone.py `# VERIFY`) without the ~80GB download.

    python -m scripts.make_fake_dataset /data/fake_kitti --scans 3
    python -m src.train task=semantic model=minkunet data.root=/data/fake_kitti \
        trainer.max_epochs=1 data.batch_size=1 data.num_workers=2

Writes real-format files so `SemanticKITTIPanoptic` parses them exactly as the real set:
  <out>/sequences/{00,08}/velodyne/{i}.bin   float32 [N,4]  (x,y,z,remission)
  <out>/sequences/{00,08}/labels/{i}.label   uint32  [N]    (inst<<16 | raw_semantic)
Seq 00 -> train split, seq 08 -> val split, so `trainer.fit` gets both loaders.
NOT for training a real model — geometry is random; only the plumbing is real.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.semantic_kitti import LEARNING_MAP, THING_TRAIN_IDS

# raw semantic ids present in the standard learning map (skip the 0/unlabeled-heavy ones a bit)
_RAW_IDS = np.array(sorted(k for k in LEARNING_MAP if k != 0), dtype=np.uint32)
# raw ids that map to a *thing* train id -> give those points a fake instance id
_THING_RAW = np.array([k for k in _RAW_IDS if LEARNING_MAP[int(k)] in THING_TRAIN_IDS], dtype=np.uint32)


def _write_scan(bin_path: Path, label_path: Path, n: int, rng: np.random.Generator) -> None:
    xyz = rng.uniform(-40, 40, size=(n, 3)).astype(np.float32)
    remission = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)
    np.concatenate([xyz, remission], axis=1).tofile(bin_path)  # [N,4] float32

    raw_sem = rng.choice(_RAW_IDS, size=n).astype(np.uint32)
    is_thing = np.isin(raw_sem, _THING_RAW)
    inst = np.where(is_thing, rng.integers(1, 15, size=n), 0).astype(np.uint32)
    label = (inst << 16) | raw_sem  # SemanticKITTI packing
    label.astype(np.uint32).tofile(label_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="dataset root (contains sequences/)")
    ap.add_argument("--scans", type=int, default=3, help="scans per sequence")
    ap.add_argument("--points", type=int, default=30000, help="points per scan")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    for seq in ("00", "08"):  # train + val
        vel = Path(args.out) / "sequences" / seq / "velodyne"
        lab = Path(args.out) / "sequences" / seq / "labels"
        vel.mkdir(parents=True, exist_ok=True)
        lab.mkdir(parents=True, exist_ok=True)
        for i in range(args.scans):
            stem = f"{i:06d}"
            _write_scan(vel / f"{stem}.bin", lab / f"{stem}.label", args.points, rng)
    print(f"wrote {args.scans} scans/seq to {args.out}/sequences/{{00,08}} "
          f"-> train on data.root={args.out}")


if __name__ == "__main__":
    main()
