"""Evaluate a checkpoint on val (seq 08): mIoU + (panoptic) PQ/PQ†/SQ/RQ, a per-class table,
and inference FPS (network-only, and end-to-end incl. offset-shift DBSCAN clustering).

    python -m src.eval ckpt=runs/best.ckpt task=semantic
    python -m src.eval ckpt=runs/best.ckpt task=panoptic   # needs vendored np_ioueval.py
"""

from __future__ import annotations

import time

import hydra
import torch
from omegaconf import DictConfig

from .data.datamodule import SemanticKITTIDataModule
from .data.semantic_kitti import CLASS_NAMES, IGNORE_ID, THING_TRAIN_IDS
from .lit_module import PanopticLit


def _to(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _print_semantic_table(iou) -> None:
    print(f"\n{'class':<16}{'IoU':>8}")
    print("-" * 24)
    for c, name in enumerate(CLASS_NAMES):
        if c == IGNORE_ID:
            continue
        print(f"{name:<16}{iou[c] * 100:>7.1f}%")


def _print_panoptic_table(pc) -> None:
    print(f"\n{'class':<16}{'':>2}{'PQ':>7}{'SQ':>7}{'RQ':>7}{'IoU':>7}")
    print("-" * 46)
    for c, name in enumerate(CLASS_NAMES):
        if c == IGNORE_ID:
            continue
        kind = "T" if c in THING_TRAIN_IDS else "S"
        print(f"{name:<16}{kind:>2}{pc['PQ'][c] * 100:>6.1f}{pc['SQ'][c] * 100:>7.1f}"
              f"{pc['RQ'][c] * 100:>7.1f}{pc['IoU'][c] * 100:>7.1f}")
    print("(T = thing, S = stuff; values in %)")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.ckpt:
        raise SystemExit("pass ckpt=path/to/best.ckpt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    panoptic = cfg.task == "panoptic"
    use_amp = "16" in str(cfg.trainer.precision) and device == "cuda"

    dm = SemanticKITTIDataModule(cfg)
    loader = dm.val_dataloader()
    model = PanopticLit.load_from_checkpoint(cfg.ckpt, cfg=cfg).to(device).eval()
    model.val_iou.reset()

    net_t, clu_t, n_scans = 0.0, 0.0, 0

    def _sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = _to(batch, device)
            _sync(); t0 = time.perf_counter()
            with torch.autocast("cuda", enabled=use_amp):
                out = model(batch)
            _sync(); net_t += time.perf_counter() - t0

            pred = out["sem_logits"].argmax(-1)
            model.val_iou.add(pred.cpu().numpy(), batch["sem"].cpu().numpy())
            if panoptic:
                t1 = time.perf_counter()
                model._accumulate_panoptic(out, batch, pred)  # offset-shift + DBSCAN (CPU)
                clu_t += time.perf_counter() - t1

            n_scans += len(batch["meta"])
            if (i + 1) % 50 == 0:
                print(f"  ...{n_scans} scans")

    iou, miou = model.val_iou.compute()
    print(f"\n=== {cfg.task} eval on val ({n_scans} scans) ===")
    print(f"mIoU: {miou * 100:.1f}%")
    if panoptic and model.pq is not None:
        s = model.pq.summary()
        print(f"PQ: {s['PQ'] * 100:.1f}  PQ†: {s['PQ_dagger'] * 100:.1f}  "
              f"SQ: {s['SQ'] * 100:.1f}  RQ: {s['RQ'] * 100:.1f}")
        _print_panoptic_table(model.pq.per_class())
    else:
        _print_semantic_table(iou)

    net_fps = n_scans / net_t if net_t else 0.0
    print(f"\nFPS (network, {cfg.trainer.precision}): {net_fps:.1f} scans/s ({net_t / n_scans * 1e3:.1f} ms/scan)")
    if panoptic:
        e2e_fps = n_scans / (net_t + clu_t) if (net_t + clu_t) else 0.0
        print(f"FPS (end-to-end + clustering): {e2e_fps:.1f} scans/s "
              f"(clustering {clu_t / n_scans * 1e3:.1f} ms/scan)")


if __name__ == "__main__":
    main()
