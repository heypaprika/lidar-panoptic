"""Hydra entrypoint. GATE 1: `python -m src.train task=semantic`."""

from __future__ import annotations

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from .data.datamodule import SemanticKITTIDataModule
from .lit_module import PanopticLit


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    dm = SemanticKITTIDataModule(cfg)
    model = PanopticLit(cfg)
    ckpt = ModelCheckpoint(monitor="val/mIoU", mode="max", save_top_k=1, filename="best")
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        devices=cfg.trainer.devices,
        callbacks=[ckpt],
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
