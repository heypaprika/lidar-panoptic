"""Sparse-voxel backbones.

GATE 1 runnable baseline: `MinkUNetBackbone` (torchsparse). Reproduce semantic mIoU with this,
then (upgrade) swap in SPVCNN's point-voxel branch by vendoring mit-han-lab/spvnas into the same
`forward(batch) -> per_point_feat [Ptot, feat_channels]` interface.

⚠️ torchsparse API note: written against torchsparse **v2.1**. If your version differs, the spots to
verify are marked `# VERIFY`: SparseTensor coords column order, spnn layer signatures, torchsparse.cat.
"""

from __future__ import annotations

import torch
from torch import nn

try:  # keep import-safe on CPU/CI without torchsparse
    import torchsparse
    import torchsparse.nn as spnn
    from torchsparse import SparseTensor
except ImportError:  # pragma: no cover
    torchsparse = None


def _block(inc: int, outc: int, stride: int = 1, transposed: bool = False) -> nn.Module:
    # VERIFY: spnn.Conv3d(in, out, kernel_size, stride, transposed=...) signature for your version
    return nn.Sequential(
        spnn.Conv3d(inc, outc, kernel_size=3, stride=stride, transposed=transposed),
        spnn.BatchNorm(outc),
        spnn.ReLU(True),
    )


class MinkUNetBackbone(nn.Module):
    """Compact sparse UNet (encoder/decoder + skips). Output: per-point features via devoxelize."""

    def __init__(self, in_channels: int = 4, feat_channels: int = 96, cr: float = 1.0):
        super().__init__()
        if torchsparse is None:
            raise ImportError("torchsparse not installed — see README setup.")
        c = [int(cr * x) for x in (32, 64, 128, 256)]
        self.stem = _block(in_channels, c[0])
        self.enc1 = _block(c[0], c[1], stride=2)
        self.enc2 = _block(c[1], c[2], stride=2)
        self.enc3 = _block(c[2], c[3], stride=2)
        self.up3 = _block(c[3], c[2], stride=2, transposed=True)
        self.red3 = _block(c[2] + c[2], c[2])
        self.up2 = _block(c[2], c[1], stride=2, transposed=True)
        self.red2 = _block(c[1] + c[1], c[1])
        self.up1 = _block(c[1], c[0], stride=2, transposed=True)
        self.red1 = _block(c[0] + c[0], feat_channels)

    def forward(self, batch: dict) -> torch.Tensor:
        # batch tensors are already on the module device (Lightning moves dict tensors).
        x = SparseTensor(feats=batch["feats"], coords=batch["coords"])  # VERIFY: coords [N,4]=(x,y,z,b)
        s = self.stem(x)
        e1, e2, e3 = self.enc1(s), None, None
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d = self.red3(torchsparse.cat([self.up3(e3), e2]))  # VERIFY: torchsparse.cat aligns by coords
        d = self.red2(torchsparse.cat([self.up2(d), e1]))
        d = self.red1(torchsparse.cat([self.up1(d), s]))
        vox_feat = d.F  # [Vtot, feat_channels], aligned to input coords order
        return vox_feat[batch["inverse"]]  # devoxelize -> [Ptot, feat_channels]


class SPVCNNBackbone(nn.Module):
    """Upgrade: SPVCNN (point-voxel). Vendor SPVCNN from mit-han-lab/spvnas into this interface."""

    def __init__(self, in_channels: int = 4, feat_channels: int = 96, cr: float = 1.0):
        super().__init__()
        raise NotImplementedError(
            "Vendor SPVCNN from mit-han-lab/spvnas (torchsparse) → forward(batch)->[Ptot,feat]."
        )

    def forward(self, batch: dict) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class DummyBackbone(nn.Module):
    """torchsparse-free per-point MLP over devoxelized features. For CI/smoke tests only
    (verifies collate/heads/losses/eval wiring without the sparse conv build)."""

    def __init__(self, in_channels: int = 4, feat_channels: int = 96, cr: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, feat_channels), nn.ReLU(True),
            nn.Linear(feat_channels, feat_channels),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        pf = batch["feats"][batch["inverse"]]  # voxel feat -> per point [Ptot, in_channels]
        return self.net(pf)


def build_backbone(cfg) -> nn.Module:
    name = cfg.model.name
    kw = dict(in_channels=cfg.model.in_channels, feat_channels=cfg.model.feat_channels, cr=cfg.model.cr)
    if name == "minkunet":
        return MinkUNetBackbone(**kw)
    if name == "spvcnn":
        return SPVCNNBackbone(**kw)
    if name == "dummy":
        return DummyBackbone(**kw)
    raise ValueError(f"unknown backbone {name}")
