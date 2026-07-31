"""Sparse-voxel backbone (spconv).

MinkUNet-style U-Net on **spconv**. We moved off torchsparse: its 2.0.0b strided conv mis-generated
output coordinates (a single stride-2 conv on clean input roughly doubled nnz and produced negative
coords), crashing deeper layers. spconv ships prebuilt CUDA wheels (no source build) and is a proven
backbone for SemanticKITTI voxel nets (Cylinder3D et al.).

Interface: forward(batch) -> per-point features [Ptot, feat_channels] (devoxelized via batch['inverse']).
"""

from __future__ import annotations

import torch
from torch import nn

try:  # keep import-safe on CPU/CI without spconv (DummyBackbone still works)
    import spconv.pytorch as spconv
    from spconv.core import ConvAlgo

    _HAS_SPCONV = True
except ImportError:  # pragma: no cover
    spconv = None
    ConvAlgo = None
    _HAS_SPCONV = False

# spconv's default implicit-GEMM kernels SIGFPE on sm_86 (RTX 3090) in this build; the Native algo
# avoids that path. (Slightly slower but correct — revisit if a newer spconv fixes implicit GEMM.)
_ALGO = ConvAlgo.Native if _HAS_SPCONV else None


def _subm(inc: int, outc: int, key: str) -> "spconv.SparseSequential":
    # submanifold conv keeps the coordinate set fixed (feature extraction, no down/upsample)
    return spconv.SparseSequential(
        spconv.SubMConv3d(inc, outc, 3, padding=1, bias=False, indice_key=key, algo=_ALGO),
        nn.BatchNorm1d(outc), nn.ReLU(True),
    )


def _down(inc: int, outc: int, key: str) -> "spconv.SparseSequential":
    return spconv.SparseSequential(
        spconv.SparseConv3d(inc, outc, 3, stride=2, padding=1, bias=False, indice_key=key, algo=_ALGO),
        nn.BatchNorm1d(outc), nn.ReLU(True),
    )


def _up(inc: int, outc: int, key: str) -> "spconv.SparseSequential":
    # inverse conv restores the exact coords/order of the matching `key` downsample -> skip-concat aligns
    return spconv.SparseSequential(
        spconv.SparseInverseConv3d(inc, outc, 3, bias=False, indice_key=key, algo=_ALGO),
        nn.BatchNorm1d(outc), nn.ReLU(True),
    )


class SpconvUNetBackbone(nn.Module):
    """Compact sparse U-Net (encoder/decoder + skips). Output: per-point features via devoxelize."""

    def __init__(self, in_channels: int = 4, feat_channels: int = 96, cr: float = 1.0):
        super().__init__()
        if not _HAS_SPCONV:
            raise ImportError("spconv not installed — pip install spconv-cu120 (see README setup).")
        c = [int(cr * x) for x in (32, 64, 128, 256)]
        self.stem = _subm(in_channels, c[0], "subm0")
        self.down1, self.enc1 = _down(c[0], c[1], "sp1"), _subm(c[1], c[1], "subm1")
        self.down2, self.enc2 = _down(c[1], c[2], "sp2"), _subm(c[2], c[2], "subm2")
        self.down3, self.enc3 = _down(c[2], c[3], "sp3"), _subm(c[3], c[3], "subm3")
        self.up3, self.red3 = _up(c[3], c[2], "sp3"), _subm(c[2] * 2, c[2], "subm2d")
        self.up2, self.red2 = _up(c[2], c[1], "sp2"), _subm(c[1] * 2, c[1], "subm1d")
        self.up1, self.red1 = _up(c[1], c[0], "sp1"), _subm(c[0] * 2, feat_channels, "subm0d")

    def forward(self, batch: dict) -> torch.Tensor:
        coords = batch["coords"].int()                    # [N,4] = (batch, x, y, z)
        bs = int(coords[:, 0].max().item()) + 1
        ss = (coords[:, 1:].amax(0) + 1).tolist()         # spatial_shape [X, Y, Z]
        x = spconv.SparseConvTensor(batch["feats"], coords, ss, bs)
        s = self.stem(x)
        e1 = self.enc1(self.down1(s))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))
        d = self.up3(e3); d = d.replace_feature(torch.cat([d.features, e2.features], 1)); d = self.red3(d)
        d = self.up2(d);  d = d.replace_feature(torch.cat([d.features, e1.features], 1)); d = self.red2(d)
        d = self.up1(d);  d = d.replace_feature(torch.cat([d.features, s.features], 1));  d = self.red1(d)
        # matched indice_keys restore stem coords/order -> d.features is in input voxel order
        return d.features[batch["inverse"]]               # devoxelize -> [Ptot, feat_channels]


class DummyBackbone(nn.Module):
    """spconv-free per-point MLP over devoxelized features. For CI/smoke tests only
    (verifies collate/heads/losses/eval wiring without the sparse-conv build)."""

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
    if name in ("minkunet", "spvcnn", "spconv"):  # MinkUNet-style U-Net on spconv
        return SpconvUNetBackbone(**kw)
    if name == "dummy":
        return DummyBackbone(**kw)
    raise ValueError(f"unknown backbone {name}")
