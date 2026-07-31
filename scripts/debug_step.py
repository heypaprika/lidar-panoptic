"""Pinpoint a training-time CUDA error: one real batch (bs=2) through forward + loss + backward,
synchronously (CUDA_LAUNCH_BLOCKING), with shape/range checks between stages.

    python -m scripts.debug_step /path/to/dataset

Isolates whether an 'illegal memory access' comes from multi-batch devoxelize (backbone out rows
!= point count), the loss (label out of range), or the spconv Native backward.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # must precede torch cuda init

import sys

import torch
from omegaconf import OmegaConf

from src.data.collate import voxelize_collate
from src.data.dataset import SemanticKITTIPanoptic
from src.data.semantic_kitti import NUM_CLASSES
from src.lit_module import PanopticLit

CFG = OmegaConf.create({
    "task": "semantic",
    "model": {"name": "minkunet", "in_channels": 4, "feat_channels": 96, "num_classes": NUM_CLASSES, "cr": 1.0},
    "loss": {"ce": 1.0, "lovasz": 1.0, "center": 1.0, "offset": 1.0, "center_sigma": 1.0},
    "cluster": {"eps": 0.6, "min_points": 20},
    "data": {"voxel": 0.05},
})


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "data/semantickitti/dataset"
    dev = "cuda"
    ds = SemanticKITTIPanoptic(root, "train")
    b = voxelize_collate([ds[0], ds[1]], voxel=0.05, in_channels=4)  # batch of 2
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    Vtot, pts = b["coords"].shape[0], b["sem"].shape[0]
    print(f"batches={int(b['coords'][:,0].max())+1} voxels(Vtot)={Vtot} points={pts} "
          f"inverse.max={int(b['inverse'].max())} (must be < Vtot)")
    assert int(b["inverse"].max()) < Vtot, "inverse index out of range vs voxel count"

    m = PanopticLit(CFG).to(dev).train()
    print("backbone forward...")
    feat = m.backbone(b); torch.cuda.synchronize()
    print(f"  backbone out rows={feat.shape[0]} (expect points={pts}) finite={torch.isfinite(feat).all().item()}")
    assert feat.shape[0] == pts, "devoxelize row count != points (multi-batch order/count mismatch)"

    out = m.heads(feat); torch.cuda.synchronize()
    print(f"  heads sem_logits={tuple(out['sem_logits'].shape)}")
    print(f"  sem label range=[{int(b['sem'].min())},{int(b['sem'].max())}] (must be within [0,{NUM_CLASSES-1}])")

    loss, logs = m._semantic_loss(out, b); torch.cuda.synchronize()
    print(f"  loss={float(loss):.4f} ce={float(logs['ce']):.4f} lovasz={float(logs['lovasz']):.4f}")
    print("backward...")
    loss.backward(); torch.cuda.synchronize()
    g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print(f"STEP OK — forward+loss+backward on bs=2, grad_sum={g:.1f}")


if __name__ == "__main__":
    main()
