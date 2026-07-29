"""Visualize one scan: semantic prediction vs panoptic (instance-colored) from a checkpoint.

    python -m scripts.viz ckpt=runs/best.ckpt viz.seq=08 viz.frame=000000        # interactive
    python -m scripts.viz ckpt=runs/best.ckpt viz.frame=000100 viz.save=demo/    # headless PNGs

Renders two clouds over the same points: colored by predicted semantic class, and by clustered
instance id (stuff/no-instance = a single background color). Uses the trained heads + offset-shift
DBSCAN, so it needs a checkpoint (and torchsparse for a real backbone).
"""

from __future__ import annotations

import os

import hydra
import torch
from omegaconf import DictConfig

from src.data.collate import voxelize_collate
from src.data.dataset import SPLITS, SemanticKITTIPanoptic
from src.lit_module import PanopticLit
from src.panoptic.cluster import panoptic_from_offsets
from src.viz import render


def _split_of(seq: str) -> str:
    for split, seqs in SPLITS.items():
        if seq in seqs:
            return split
    raise ValueError(f"seq {seq} not in any split")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.ckpt:
        raise SystemExit("pass ckpt=path/to/best.ckpt")
    seq, frame = str(cfg.viz.seq), str(cfg.viz.frame)
    ds = SemanticKITTIPanoptic(cfg.data.root, _split_of(seq))
    try:
        sample = ds[ds.frames.index((seq, frame))]
    except ValueError:
        raise SystemExit(f"scan {seq}/{frame} not found under {cfg.data.root}")

    batch = voxelize_collate([sample], voxel=cfg.data.voxel, in_channels=cfg.model.in_channels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    model = PanopticLit.load_from_checkpoint(cfg.ckpt, cfg=cfg).to(device).eval()
    with torch.no_grad():
        out = model(batch)

    xyz = batch["xyz"].cpu().numpy()
    sem_pred = out["sem_logits"].argmax(-1).cpu().numpy()
    inst_pred = panoptic_from_offsets(
        xyz, sem_pred, out["offset"].cpu().numpy(),
        eps=cfg.cluster.eps, min_points=cfg.cluster.min_points,
    )
    print(f"{seq}/{frame}: {len(xyz)} pts, {int(inst_pred.max())} instances")

    if cfg.viz.save:
        os.makedirs(cfg.viz.save, exist_ok=True)
        base = os.path.join(cfg.viz.save, f"{seq}_{frame}")
        render.save(xyz, sem_pred, f"{base}_semantic.png")
        render.save(xyz, inst_pred, f"{base}_panoptic.png")
        print(f"wrote {base}_semantic.png / _panoptic.png")
    else:
        render.show(xyz, sem_pred)   # semantic
        render.show(xyz, inst_pred)  # panoptic (instance-colored)


if __name__ == "__main__":
    main()
