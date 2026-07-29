"""LightningModule: backbone -> panoptic heads -> losses.

task='semantic' (GATE 1) is fully wired: CE + Lovász, val mIoU.
task='panoptic' (GATE 2) adds center(MSE)+offset(L1); instance targets are the TODO below.
"""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .data.semantic_kitti import IGNORE_ID, NUM_CLASSES, THING_TRAIN_IDS
from .losses import lovasz_softmax
from .metrics import IoUMeter
from .models.backbone import build_backbone
from .models.heads import PanopticHeads


class PanopticLit(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.task = cfg.task
        self.backbone = build_backbone(cfg)
        self.heads = PanopticHeads(cfg.model.feat_channels, NUM_CLASSES)
        self.register_buffer("thing", torch.tensor(sorted(THING_TRAIN_IDS)))
        self.val_iou = IoUMeter(NUM_CLASSES, ignore=IGNORE_ID)

    def forward(self, batch) -> dict:
        return self.heads(self.backbone(batch))

    def _semantic_loss(self, out, batch) -> tuple[torch.Tensor, dict]:
        sem = batch["sem"]
        ce = F.cross_entropy(out["sem_logits"], sem, ignore_index=IGNORE_ID)
        lov = lovasz_softmax(out["sem_logits"].softmax(-1), sem, ignore=IGNORE_ID)
        loss = self.cfg.loss.ce * ce + self.cfg.loss.lovasz * lov
        return loss, {"ce": ce.detach(), "lovasz": lov.detach()}

    def _instance_loss(self, out, batch):
        # TODO(gate2): per-thing-instance centroid -> offset_gt = centroid - xyz;
        #   center_gt = exp(-||offset||^2 / (2σ²)); MSE(center), L1(offset) on thing points.
        raise NotImplementedError("GATE 2: instance targets (see DESIGN §2).")

    def training_step(self, batch, _):
        out = self(batch)
        loss, logs = self._semantic_loss(out, batch)
        if self.task == "panoptic":
            loss = loss + self._instance_loss(out, batch)
        self.log_dict({f"train/{k}": v for k, v in {**logs, "loss": loss}.items()}, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        out = self(batch)
        pred = out["sem_logits"].argmax(-1)
        self.val_iou.add(pred.cpu().numpy(), batch["sem"].cpu().numpy())

    def on_validation_epoch_end(self):
        iou, miou = self.val_iou.compute()
        self.log("val/mIoU", miou, prog_bar=True)
        self.val_iou.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
