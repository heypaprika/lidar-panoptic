"""Run the real backbone stage-by-stage on one real scan to locate the torchsparse crash.

    CUDA_LAUNCH_BLOCKING=1 python -m scripts.debug_backbone /path/to/dataset [split]

Prints nnz / coord range / tensor-stride at stem, enc1, enc2, then attempts enc3 (where the
'invalid configuration argument' crash happens) so we see the exact tensor that triggers it.
Also confirms the collate fix is present (coord min should be >= 0).
"""

from __future__ import annotations

import sys

import torch
import torchsparse
from torchsparse import SparseTensor

from src.data.collate import voxelize_collate
from src.data.dataset import SemanticKITTIPanoptic
from src.models.backbone import MinkUNetBackbone


def _stride(t):
    for a in ("stride", "s", "tensor_stride"):
        v = getattr(t, a, None)
        if v is not None:
            return v
    return "?"


def _show(t, name: str) -> None:
    c = t.C[:, :3]
    print(f"{name:6s} nnz={t.F.shape[0]:>7d}  Cshape={tuple(t.C.shape)}  "
          f"stride={_stride(t)}  cmin={int(c.min())} cmax={int(c.max())}")


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "data/semantickitti/dataset"
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    print("torchsparse", torchsparse.__version__)

    ds = SemanticKITTIPanoptic(root, split)
    b = voxelize_collate([ds[0]], voxel=0.05, in_channels=4)
    b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    print(f"scan points={b['xyz'].shape[0]} voxels={b['coords'].shape[0]} "
          f"coord min={int(b['coords'][:, :3].min())} max={int(b['coords'][:, :3].max())} "
          f"(min>=0 means collate fix present)")

    m = MinkUNetBackbone().cuda().eval()
    x = SparseTensor(feats=b["feats"], coords=b["coords"])
    with torch.no_grad():
        s0 = m.stem(x); torch.cuda.synchronize(); _show(s0, "stem")
        e1 = m.enc1(s0); torch.cuda.synchronize(); _show(e1, "enc1")
        e2 = m.enc2(e1); torch.cuda.synchronize(); _show(e2, "enc2")
        print("-> attempting enc3 (crash point) ...")
        e3 = m.enc3(e2); torch.cuda.synchronize(); _show(e3, "enc3")
    print("BACKBONE OK — all encoder stages passed")


if __name__ == "__main__":
    main()
