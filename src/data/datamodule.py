"""LightningDataModule for SemanticKITTI panoptic."""

from __future__ import annotations

from functools import partial

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .collate import voxelize_collate
from .dataset import SemanticKITTIPanoptic


class SemanticKITTIDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._collate = partial(
            voxelize_collate, voxel=cfg.data.voxel, in_channels=cfg.model.in_channels
        )

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            SemanticKITTIPanoptic(self.cfg.data.root, split),
            batch_size=self.cfg.data.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.data.num_workers,
            collate_fn=self._collate,
            pin_memory=True,
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", False)
