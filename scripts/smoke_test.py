"""Synthetic end-to-end smoke test — no dataset, no torchsparse required.

Verifies the glue we own (collate -> dummy backbone -> panoptic heads -> semantic loss ->
IoU meter -> backward) on the real torch/pl/GPU stack. The torchsparse MinkUNet path is verified
separately on a box with the sparse-conv build (see backbone.py `# VERIFY`).

    python -m scripts.smoke_test
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.collate import voxelize_collate
from src.data.semantic_kitti import NUM_CLASSES, THING_TRAIN_IDS
from src.lit_module import PanopticLit
from src.panoptic.cluster import panoptic_from_offsets

CFG = OmegaConf.create(
    {
        "task": "semantic",
        "model": {"name": "dummy", "in_channels": 4, "feat_channels": 96, "num_classes": NUM_CLASSES, "cr": 1.0},
        "loss": {"ce": 1.0, "lovasz": 1.0, "center": 1.0, "offset": 1.0, "center_sigma": 1.0},
        "cluster": {"eps": 0.6, "min_points": 20},
        "data": {"voxel": 0.05},
    }
)


def _fake_scan(seq: str, frame: str, n: int = 20000) -> dict:
    rng = np.random.default_rng(abs(hash((seq, frame))) % (2**32))
    xyz = rng.uniform(-50, 50, size=(n, 3)).astype(np.float32)
    feat = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)  # remission
    sem = rng.integers(0, NUM_CLASSES, size=n).astype(np.int64)  # includes ignore=0
    inst = np.where(np.isin(sem, list(THING_TRAIN_IDS)), rng.integers(1, 40, size=n), 0).astype(np.int64)
    return {"seq": seq, "frame": frame, "xyz": xyz, "feat": feat, "sem": sem, "inst": inst}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = [_fake_scan("00", "000000"), _fake_scan("00", "000001")]
    batch = voxelize_collate(samples, voxel=CFG.data.voxel, in_channels=CFG.model.in_channels)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    print(f"[shapes] coords={tuple(batch['coords'].shape)} feats={tuple(batch['feats'].shape)} "
          f"inverse={tuple(batch['inverse'].shape)} sem={tuple(batch['sem'].shape)}")
    assert batch["inverse"].max().item() + 1 == batch["coords"].shape[0], "inverse must index every voxel"

    model = PanopticLit(CFG).to(device)
    model.train()

    # forward + semantic loss + backward (no Trainer needed)
    out = model(batch)
    assert out["sem_logits"].shape == (batch["sem"].shape[0], NUM_CLASSES), out["sem_logits"].shape
    loss, logs = model._semantic_loss(out, batch)
    loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"[train] loss={loss.item():.4f} ce={logs['ce']:.4f} lovasz={logs['lovasz']:.4f} grad_sum={grad:.2f}")
    assert torch.isfinite(loss) and grad > 0, "loss must be finite and produce gradients"

    # validation: IoU meter accumulation + compute
    model.eval()
    with torch.no_grad():
        out = model(batch)
        pred = out["sem_logits"].argmax(-1)
        model.val_iou.add(pred.cpu().numpy(), batch["sem"].cpu().numpy())
    iou, miou = model.val_iou.compute()
    print(f"[val] mIoU={miou:.4f} (random preds, sanity only) per_class_len={len(iou)}")
    assert len(iou) == NUM_CLASSES and np.isfinite(miou)

    # ---- GATE 2: panoptic path (instance targets + loss + clustering; PQ needs vendored eval) ----
    pcfg = OmegaConf.merge(CFG, {"task": "panoptic"})
    pmodel = PanopticLit(pcfg).to(device)
    pmodel.train()
    out = pmodel(batch)
    off_gt, ctr_gt, thing = pmodel._instance_targets(batch)
    assert off_gt.shape == (batch["sem"].shape[0], 3) and ctr_gt.shape == batch["sem"].shape
    assert thing.any(), "fake scan should contain thing points"
    # non-thing center_gt is exactly 0; some thing points fire (>0). (Random scans scatter instance
    # points across the whole volume, so far points underflow to 0 — real instances are compact.)
    assert ctr_gt[~thing].abs().sum() == 0 and ctr_gt[thing].max() > 0, "center_gt confined to things"
    sloss, _ = pmodel._semantic_loss(out, batch)
    iloss, ilogs = pmodel._instance_loss(out, batch)
    total = sloss + iloss
    total.backward()
    pgrad = sum(p.grad.abs().sum().item() for p in pmodel.parameters() if p.grad is not None)
    print(f"[panoptic] center={ilogs['center']:.4f} offset={ilogs['offset']:.4f} "
          f"thing_pts={int(thing.sum())} grad_sum={pgrad:.2f}")
    assert torch.isfinite(total) and pgrad > 0

    # offset-shift clustering on one scan -> instance ids
    m = (batch["pbatch"] == 0).cpu().numpy()
    inst_p = panoptic_from_offsets(
        batch["xyz"].cpu().numpy()[m], batch["sem"].cpu().numpy()[m],
        out["offset"].detach().cpu().numpy()[m], eps=pcfg.cluster.eps, min_points=pcfg.cluster.min_points,
    )
    print(f"[cluster] scan0 -> {int(inst_p.max())} instances over {int((inst_p > 0).sum())} thing pts")
    assert inst_p.shape == (m.sum(),)

    print("\nSMOKE OK — semantic + panoptic(instance loss/targets/cluster) verified on", device)


if __name__ == "__main__":
    main()
