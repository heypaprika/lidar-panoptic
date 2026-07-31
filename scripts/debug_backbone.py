"""Sanity-run the spconv backbone on one real scan: full forward + output shape + timing.

    python -m scripts.debug_backbone /path/to/dataset [split]

Confirms the sparse-conv path works end-to-end on real KITTI geometry (the torchsparse strided-conv
bug is gone with spconv). Reports voxel count, per-point output shape, and forward latency.
"""

from __future__ import annotations

import sys
import time

import torch

from src.data.collate import voxelize_collate
from src.data.dataset import SemanticKITTIPanoptic
from src.models.backbone import SpconvUNetBackbone


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "data/semantickitti/dataset"
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SemanticKITTIPanoptic(root, split)
    b = voxelize_collate([ds[0]], voxel=0.05, in_channels=4)
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    print(f"scan points={b['xyz'].shape[0]} voxels={b['coords'].shape[0]} "
          f"coord[min={int(b['coords'][:, 1:4].min())}, max={int(b['coords'][:, 1:4].max())}]")

    m = SpconvUNetBackbone().to(dev).eval()
    with torch.no_grad():
        out = m(b)  # warm-up (builds spconv kernels)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = m(b)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    assert out.shape[0] == b["xyz"].shape[0], (out.shape, b["xyz"].shape)  # per-point features
    print(f"per-point feat: {tuple(out.shape)}  forward={dt * 1e3:.0f} ms  finite={torch.isfinite(out).all().item()}")
    print("BACKBONE OK — spconv U-Net ran end-to-end on a real scan")


if __name__ == "__main__":
    main()
