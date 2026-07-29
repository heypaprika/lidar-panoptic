"""Lovász-softmax loss (Berman et al., 2018) — flat/point version.

Points are already flattened ([N, C] probs, [N] labels), so we use the "flat" form directly.
Ignore-labeled points are dropped; the ignore *class* is skipped in the per-class loop.
"""

from __future__ import annotations

import torch


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
    return jaccard


def lovasz_softmax(probas: torch.Tensor, labels: torch.Tensor, ignore: int = 0) -> torch.Tensor:
    """probas [N, C] (softmax), labels [N] in [0, C). Returns scalar."""
    valid = labels != ignore
    probas, labels = probas[valid], labels[valid]
    if probas.numel() == 0:
        return probas.sum() * 0.0
    losses = []
    for c in range(probas.size(1)):
        if c == ignore:
            continue
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()
