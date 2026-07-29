"""Hydra entrypoint. `python -m src.train [overrides]`."""

from __future__ import annotations

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from .lit_module import PanopticLit


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    # TODO(gate1): build datamodule (SemanticKITTIPanoptic + torchsparse collate) from cfg.data
    model = PanopticLit(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        devices=cfg.trainer.devices,
    )
    # trainer.fit(model, datamodule=dm)
    raise SystemExit("Wire the datamodule (GATE 1) then trainer.fit(). See TASKS.md.")


if __name__ == "__main__":
    main()
