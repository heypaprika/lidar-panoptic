"""Evaluate a checkpoint on val (seq 08). Reuses the LightningModule's validation loop, so it
reports val/mIoU and — for task=panoptic — val/PQ, PQ†, SQ, RQ (official evaluator).

    python -m src.eval ckpt=runs/best.ckpt task=semantic
    python -m src.eval ckpt=runs/best.ckpt task=panoptic   # needs vendored np_ioueval.py
"""

from __future__ import annotations

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from .data.datamodule import SemanticKITTIDataModule
from .lit_module import PanopticLit


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.ckpt:
        raise SystemExit("pass ckpt=path/to/best.ckpt")
    dm = SemanticKITTIDataModule(cfg)
    # cfg= override lets you eval a semantic-trained ckpt under task=panoptic (heads are present).
    model = PanopticLit.load_from_checkpoint(cfg.ckpt, cfg=cfg)
    trainer = pl.Trainer(
        devices=cfg.trainer.devices, precision=cfg.trainer.precision, logger=False
    )
    trainer.validate(model, datamodule=dm)


if __name__ == "__main__":
    main()
