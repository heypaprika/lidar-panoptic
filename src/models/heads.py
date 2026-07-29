"""Panoptic heads on top of a per-point backbone feature.

Bottom-up (Panoptic-DeepLab / DS-Net style):
  - SemanticHead: per-point class logits.
  - CenterHead:   per-point centerness in [0,1] (thing-ness heatmap proxy on points).
  - OffsetHead:   per-point 3D offset toward its instance center.
Backbone-agnostic: takes [N, C] point features (dense torch.Tensor after backbone devoxelize).
"""

from __future__ import annotations

import torch
from torch import nn


def _mlp(in_c: int, hidden: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_c, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, out_c),
    )


class SemanticHead(nn.Module):
    def __init__(self, in_c: int, num_classes: int, hidden: int = 128):
        super().__init__()
        self.mlp = _mlp(in_c, hidden, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:  # [N, num_classes]
        return self.mlp(feat)


class CenterHead(nn.Module):
    def __init__(self, in_c: int, hidden: int = 128):
        super().__init__()
        self.mlp = _mlp(in_c, hidden, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:  # [N] centerness in [0,1]
        return torch.sigmoid(self.mlp(feat)).squeeze(-1)


class OffsetHead(nn.Module):
    def __init__(self, in_c: int, hidden: int = 128):
        super().__init__()
        self.mlp = _mlp(in_c, hidden, 3)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:  # [N, 3] offset to instance center
        return self.mlp(feat)


class PanopticHeads(nn.Module):
    def __init__(self, in_c: int, num_classes: int):
        super().__init__()
        self.semantic = SemanticHead(in_c, num_classes)
        self.center = CenterHead(in_c)
        self.offset = OffsetHead(in_c)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "sem_logits": self.semantic(feat),
            "center": self.center(feat),
            "offset": self.offset(feat),
        }
