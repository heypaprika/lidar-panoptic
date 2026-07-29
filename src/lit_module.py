"""LightningModule wiring: SPVCNN backbone -> panoptic heads -> losses.

Skeleton — backbone.forward and the sparse collation are the GATE-1 plug-in (src/models/backbone.py).
Losses: CE + Lovász-softmax (semantic), MSE (center), L1 (offset on thing points).
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from .data.semantic_kitti import IGNORE_ID, NUM_CLASSES, THING_TRAIN_IDS
from .models.backbone import SPVCNNBackbone
from .models.heads import PanopticHeads

# TODO: from .losses import lovasz_softmax   # vendor the standard Lovász-softmax impl


class PanopticLit(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.backbone = SPVCNNBackbone(
            in_channels=cfg.model.in_channels,
            feat_channels=cfg.model.feat_channels,
            cr=cfg.model.cr,
        )
        self.heads = PanopticHeads(cfg.model.feat_channels, NUM_CLASSES)
        self.thing = torch.tensor(sorted(THING_TRAIN_IDS))

    def forward(self, batch) -> dict:
        feat = self.backbone(batch)          # [N, feat_channels]
        return self.heads(feat)

    def _loss(self, out: dict, batch: dict) -> dict:
        w = self.cfg.loss
        sem, inst, xyz = batch["sem"], batch["inst"], batch["xyz"]
        ce = F.cross_entropy(out["sem_logits"], sem, ignore_index=IGNORE_ID)
        # lov = lovasz_softmax(out["sem_logits"].softmax(-1), sem, ignore=IGNORE_ID)  # TODO
        thing_mask = torch.isin(sem, self.thing.to(sem.device))
        center_gt, offset_gt = self._instance_targets(xyz, inst, thing_mask)  # TODO helper
        center = F.mse_loss(out["center"][thing_mask], center_gt[thing_mask]) if thing_mask.any() else 0.0
        offset = F.l1_loss(out["offset"][thing_mask], offset_gt[thing_mask]) if thing_mask.any() else 0.0
        total = w.ce * ce + w.center * center + w.offset * offset  # + w.lovasz * lov
        return {"loss": total, "ce": ce, "center": center, "offset": offset}

    def _instance_targets(self, xyz, inst, thing_mask):
        # TODO(gate2): per-thing-instance centroid -> offset_gt = centroid - xyz;
        #   center_gt = exp(-||offset||^2 / (2 sigma^2))  (Gaussian centerness). See DESIGN §2.
        raise NotImplementedError

    def training_step(self, batch, _):
        losses = self._loss(self(batch), batch)
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, prog_bar=True)
        return losses["loss"]

    def validation_step(self, batch, _):
        # TODO: cluster -> PanopticScorer.add(...) and log PQ/mIoU at epoch end
        self.log_dict({f"val/{k}": v for k, v in self._loss(self(batch), batch).items()})

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
