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
from .panoptic.cluster import panoptic_from_offsets


class PanopticLit(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.task = cfg.task
        self.backbone = build_backbone(cfg)
        self.heads = PanopticHeads(cfg.model.feat_channels, NUM_CLASSES)
        self.register_buffer("thing", torch.tensor(sorted(THING_TRAIN_IDS)))
        self.sigma = float(getattr(cfg.loss, "center_sigma", 1.0))
        self.val_iou = IoUMeter(NUM_CLASSES, ignore=IGNORE_ID)
        # official PQ evaluator is heavy + needs the vendored eval; only build it for panoptic.
        self.pq = None

    def forward(self, batch) -> dict:
        return self.heads(self.backbone(batch))

    def _semantic_loss(self, out, batch) -> tuple[torch.Tensor, dict]:
        sem = batch["sem"]
        ce = F.cross_entropy(out["sem_logits"], sem, ignore_index=IGNORE_ID)
        lov = lovasz_softmax(out["sem_logits"].softmax(-1), sem, ignore=IGNORE_ID)
        loss = self.cfg.loss.ce * ce + self.cfg.loss.lovasz * lov
        return loss, {"ce": ce.detach(), "lovasz": lov.detach()}

    def _instance_targets(self, batch):
        """Per-point instance regression targets (thing points only):
          offset_gt = instance_centroid(xyz) - xyz     (0 for stuff/ignore)
          center_gt = exp(-||offset_gt||^2 / 2σ²)       (0 for stuff/ignore)
        Instances are keyed by (scan, instance-id) so ids reused across scans don't merge.
        """
        xyz, sem, inst, pbatch = batch["xyz"], batch["sem"], batch["inst"], batch["pbatch"]
        thing_mask = (inst > 0) & torch.isin(sem, self.thing.to(sem.device))
        offset_gt = torch.zeros_like(xyz)
        center_gt = torch.zeros(xyz.shape[0], device=xyz.device)
        if thing_mask.any():
            b, ii, pts = pbatch[thing_mask], inst[thing_mask], xyz[thing_mask]
            key = b * (int(ii.max()) + 1) + ii                       # unique per (scan, instance)
            _, inv = torch.unique(key, return_inverse=True)
            g = int(inv.max()) + 1
            sums = torch.zeros(g, 3, device=xyz.device).index_add_(0, inv, pts)
            cnts = torch.zeros(g, device=xyz.device).index_add_(0, inv, torch.ones_like(inv, dtype=xyz.dtype))
            off = sums[inv] / cnts[inv].unsqueeze(1) - pts           # centroid - point
            offset_gt[thing_mask] = off
            center_gt[thing_mask] = torch.exp(-(off * off).sum(-1) / (2 * self.sigma**2))
        return offset_gt, center_gt, thing_mask

    def _instance_loss(self, out, batch) -> tuple[torch.Tensor, dict]:
        offset_gt, center_gt, thing_mask = self._instance_targets(batch)
        center_loss = F.mse_loss(out["center"], center_gt)           # heatmap over all points
        if thing_mask.any():                                          # offset only on thing points
            offset_loss = F.l1_loss(out["offset"][thing_mask], offset_gt[thing_mask])
        else:
            offset_loss = out["offset"].sum() * 0.0                  # keep graph, zero contribution
        loss = self.cfg.loss.center * center_loss + self.cfg.loss.offset * offset_loss
        return loss, {"center": center_loss.detach(), "offset": offset_loss.detach()}

    def training_step(self, batch, _):
        out = self(batch)
        loss, logs = self._semantic_loss(out, batch)
        if self.task == "panoptic":
            iloss, ilogs = self._instance_loss(out, batch)
            loss, logs = loss + iloss, {**logs, **ilogs}
        self.log_dict({f"train/{k}": v for k, v in {**logs, "loss": loss}.items()}, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        out = self(batch)
        pred = out["sem_logits"].argmax(-1)
        self.val_iou.add(pred.cpu().numpy(), batch["sem"].cpu().numpy())
        if self.task == "panoptic":
            self._accumulate_panoptic(out, batch, pred)

    def _accumulate_panoptic(self, out, batch, pred) -> None:
        if self.pq is None:
            from .panoptic.pq import PanopticScorer
            self.pq = PanopticScorer(min_points=self.cfg.cluster.min_points)
        xyz = batch["xyz"].cpu().numpy()
        offset = out["offset"].detach().cpu().numpy()
        sem_p, sem_g = pred.cpu().numpy(), batch["sem"].cpu().numpy()
        inst_g, pbatch = batch["inst"].cpu().numpy(), batch["pbatch"].cpu().numpy()
        for b in np.unique(pbatch):                                   # PQ is per-scan
            m = pbatch == b
            inst_p = panoptic_from_offsets(
                xyz[m], sem_p[m], offset[m],
                eps=self.cfg.cluster.eps, min_points=self.cfg.cluster.min_points,
            )
            self.pq.add(sem_p[m], inst_p, sem_g[m], inst_g[m])

    def on_validation_epoch_end(self):
        _, miou = self.val_iou.compute()
        self.log("val/mIoU", miou, prog_bar=True)
        self.val_iou.reset()
        if self.task == "panoptic" and self.pq is not None:
            s = self.pq.summary()
            self.log_dict({f"val/{k}": v for k, v in s.items()}, prog_bar=True)
            self.pq.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
