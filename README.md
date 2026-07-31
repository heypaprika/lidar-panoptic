# Lightweight Panoptic Segmentation on Sparse Point Clouds

LiDAR **panoptic segmentation** on SemanticKITTI: a sparse-voxel **semantic** backbone
(MinkUNet-style U-Net on **spconv**) extended with lightweight **center + offset** instance heads
(bottom-up, Panoptic-DeepLab / DS-Net style). Semantic backbone → per-point offset to instance
center + center heatmap → offset-shift & cluster → **panoptic output**, scored with the **official**
Panoptic Quality (PQ) and mIoU.

<!-- DEMO: after `scripts.viz` writes demo/, put the hero image here (reviewers see this first):
![semantic vs panoptic](demo/08_000100_panoptic.png)
-->

> Goal: demonstrate the jump from *"did semantic segmentation"* to *"extended a semantic backbone to
> panoptic end-to-end — with metrics, ablations, and reproducible research tooling."*

## Results (val seq 08)
Reduced setting (labeled below) — the honest tradeoff for a short compute budget; the reference row
is the published ballpark to compare against.

| Setting | mIoU | PQ | PQ† | SQ | RQ | FPS |
|---|---|---|---|---|---|---|
| semantic (spconv MinkUNet) | _tbd_ | — | — | — | — | _tbd_ |
| + center/offset panoptic | — | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| _reference (published, approx)_ | _~63_ | _~55–58_ | — | — | — | — |

> Setting: <sequences / voxel / epochs>. Gap to published is primarily epochs + voxel resolution,
> not method (fill the exact delta once measured). Reference: SPVCNN mIoU ≈ 63 (mit-han-lab/spvnas);
> bottom-up panoptic PQ ≈ 55–58 (DS-Net, Panoptic-PolarNet). Ablations: [`ablations.md`](ablations.md).

## Why this design
- **Reuse, don't rebuild**: reproduce a semantic baseline first (a go/no-go gate), then add instance
  heads on the same backbone. Gates in `TASKS.md`, math in `DESIGN.md`.
- **Center + offset, not embedding + MeanShift**: bottom-up regression is dense, well-posed, and
  stable to train → reaches non-trivial PQ fast; metric-learning embeddings are margin/bandwidth
  sensitive → kept as an *ablation* (A1) to show the alternative is understood.
- **Official eval, no hand-rolled PQ**: PQ / PQ† / SQ / RQ via SemanticKITTI's evaluator; the >0.5-IoU
  matching and void handling are subtle and easy to get wrong.
- **Latency reported**: panoptic is only useful if it runs — eval prints network + end-to-end FPS.

## Implemented here vs adapted
Written for this repo (the engineering being demonstrated):
- SemanticKITTI dataset + label remap, range crop, and pure-numpy voxelize/collate (version-robust,
  batch-first coords, non-negative shift) — `src/data/`.
- spconv MinkUNet U-Net wired for per-point output (devoxelize), Native algo — `src/models/backbone.py`.
- Semantic (CE + Lovász) and **instance targets/losses** (per-(scan,inst) centroid → offset_gt,
  Gaussian center_gt; MSE + masked L1) — `src/lit_module.py`, `src/losses.py`.
- Offset-shift **DBSCAN** clustering → panoptic merge — `src/panoptic/cluster.py`.
- PQ **adapter** over the official evaluator, incl. PQ† — `src/panoptic/pq.py`.
- Eval with per-class table + FPS, Open3D viz, Hydra/Lightning/Docker infra, gate-based plan.

Adapted / external (dependencies, not claimed as original):
- **spconv** (sparse conv kernels); **PRBonn/semantic-kitti-api** `PanopticEval` (PQ math, vendored).
- Architecture patterns: Panoptic-DeepLab / DS-Net (center+offset); MinkUNet (backbone shape).
- SemanticKITTI `learning_map` (raw→train class ids).

## Status
Pipeline verified end-to-end and **training converges** on real data (semantic mIoU climbing on
val 08). Synthetic smoke test (`scripts/smoke_test.py`) and a real-scan backbone check
(`scripts/debug_backbone.py`) both pass. Remaining: finish the reduced-setting runs and fill the
numbers above (GATE 1 mIoU → GATE 2 PQ).

## Setup
Training targets a **rented cloud GPU** (24GB+ VRAM, ~170GB disk). spconv ships prebuilt wheels — no
source build.

```bash
# A) Docker:  docker build -f docker/Dockerfile -t panoptic .
# B) Bare CUDA box (sudo):
bash scripts/setup_cloud.sh                          # pip deps incl. spconv wheel + smoke test
bash scripts/download_semantickitti.sh /data/semantickitti   # ~80GB velodyne+labels
#   -> set configs/data/semantickitti.yaml  root: /data/semantickitti/dataset
```
No GPU/data? `PYTHONPATH=. python -m scripts.smoke_test` verifies the pipeline synthetically.

## Run
```bash
DATA=/data/semantickitti/dataset
python -m src.train task=semantic model=minkunet data.root=$DATA          # GATE 1: mIoU
wget -O src/panoptic/np_ioueval.py \
  https://raw.githubusercontent.com/PRBonn/semantic-kitti-api/master/auxiliary/np_ioueval.py
python -m src.train task=panoptic model=minkunet data.root=$DATA          # GATE 2: +center/offset
python -m src.eval  ckpt=<best.ckpt> task=panoptic data.root=$DATA        # PQ/mIoU + per-class + FPS
python -m scripts.viz ckpt=<best.ckpt> viz.frame=000100 viz.save=demo/ data.root=$DATA

# reduced setting for a short budget (faster GATE numbers; label it in Results):
python -m src.train task=semantic model=minkunet data.root=$DATA \
    data.voxel=0.10 trainer.max_epochs=15 trainer.limit_train_batches=0.5
```

## Layout
```
configs/      Hydra (data / model / trainer)
src/data/     SemanticKITTI dataset + label maps + voxelize/collate
src/models/   spconv backbone + semantic/center/offset heads
src/panoptic/ offset-shift clustering + official-PQ adapter
src/viz/      Open3D rendering            scripts/  train/eval/viz + smoke/debug helpers
DESIGN.md     decisions, math & tradeoffs   TASKS.md  plan w/ go/no-go gates   ablations.md
```
