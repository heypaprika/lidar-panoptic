"""Hydra entrypoint. GATE 1: `python -m src.train task=semantic`."""

from __future__ import annotations

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from .data.datamodule import SemanticKITTIDataModule
from .lit_module import PanopticLit


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")  # use 3090 tensor cores for fp32 matmuls
    dm = SemanticKITTIDataModule(cfg)
    model = PanopticLit(cfg)

    out_dir = HydraConfig.get().runtime.output_dir  # ckpts land here (Hydra chdir's away from repo)
    ckpt = ModelCheckpoint(
        dirpath=out_dir, monitor="val/mIoU", mode="max",
        save_top_k=1, save_last=True, filename="best",
    )
    # co-locate metrics.csv with the checkpoints (Hydra 1.2+ doesn't chdir, so the default logger
    # would otherwise write to <cwd>/lightning_logs instead of the run's output dir).
    logger = CSVLogger(save_dir=out_dir, name="", version="")
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        devices=cfg.trainer.devices,
        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
        check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch", 1),
        logger=logger,
        callbacks=[ckpt, LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(model, datamodule=dm)

    print(f"\nbest  ckpt: {ckpt.best_model_path}  (val/mIoU={ckpt.best_model_score})")
    print(f"metrics:    {logger.log_dir}/metrics.csv")
    print(f"last  ckpt: {ckpt.last_model_path}")
    print(f"eval it:    python -m src.eval ckpt={ckpt.best_model_path} task={cfg.task} data.root={cfg.data.root}")


if __name__ == "__main__":
    main()
