"""SPVCNN backbone wrapper (torchsparse).

GATE 1 task: adapt the SPVCNN implementation + SemanticKITTI voxelization/collation from
`mit-han-lab/spvnas` (Apache-2.0). Keep the interface below so heads stay decoupled:

    backbone(coords_int, feats, batch) -> per_point_feat [N_total, feat_channels]

Do NOT rebuild the sparse UNet from scratch — reproduce the published SPVCNN semantic mIoU first,
then attach `PanopticHeads`.
"""

from __future__ import annotations

import torch
from torch import nn


class SPVCNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 4, feat_channels: int = 96, cr: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.cr = cr
        # TODO(gate1): build/adapt SPVCNN (torchsparse) here; expose devoxelized per-point feat.
        raise NotImplementedError(
            "Adapt SPVCNN from mit-han-lab/spvnas (torchsparse). See DESIGN.md §2-3."
        )

    def forward(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError
