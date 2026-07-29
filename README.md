# Lightweight Panoptic Segmentation on Sparse Point Clouds

LiDAR **panoptic segmentation** on SemanticKITTI, built by extending a sparse-voxel
**semantic** backbone (SPVCNN / torchsparse) with lightweight **center + offset** instance
heads (Panoptic-DeepLab / DS-Net style, bottom-up). Semantic backbone → per-point offset to
instance center + center heatmap → shift & cluster → **panoptic output**, evaluated with the
official **Panoptic Quality (PQ)** and **mIoU**.

> Goal: show the jump from *"did semantic segmentation"* to *"extended a semantic backbone to
> panoptic, end-to-end, with metrics, ablations, and reproducible research tooling."*

## Why this design
- **Reuse, don't rebuild**: reproduce a known SPVCNN semantic baseline first (go/no-go gate),
  then add instance heads. See `DESIGN.md`.
- **Center + offset (not embedding + MeanShift)**: bottom-up regression is far more stable to
  train and to reach a non-trivial PQ; embedding + clustering is kept as an *ablation*.
- **Official eval**: PQ / PQ† / SQ / RQ via SemanticKITTI's panoptic evaluator; mIoU for semantic.

## Status
Code complete through Week 5 (data · MinkUNet/SPVCNN backbone · semantic+instance heads · losses ·
offset-shift DBSCAN · official-PQ adapter · eval with per-class table + FPS · Open3D viz). The full
pipeline is verified on synthetic data via `scripts/smoke_test.py` (torch 2.4 + CUDA). Remaining is
execution on a rented cloud GPU: torchsparse `# VERIFY`, reproduce mIoU (GATE 1), PQ (GATE 2). See
`TASKS.md` for gates and `DESIGN.md` for the math.

## Results (fill in on the cloud run)
Numbers below are placeholders; the reference row is the published ballpark to compare against
(val seq 08). A **reduced setting** (subsequence / fewer epochs / lower res) is fine if labeled.

| Setting | mIoU | PQ | PQ† | SQ | RQ | FPS |
|---|---|---|---|---|---|---|
| SPVCNN semantic (repro) | — | — | — | — | — | — |
| + center/offset panoptic (GATE 2) | — | — | — | — | — | — |
| _reference (published, approx)_ | _~63_ | _~55–58_ | — | — | — | — |

Reference points: SPVCNN semantic mIoU ≈ 63 (mit-han-lab/spvnas); bottom-up panoptic PQ ≈ 55–58
(DS-Net, Panoptic-PolarNet). Ablations and their tables live in [`ablations.md`](ablations.md).

## Setup

Training targets a **rented cloud GPU** (24GB+ VRAM, ~170GB disk). Two paths:

```bash
# A) Docker (reproducible):
docker build -f docker/Dockerfile -t panoptic .

# B) Bare box with sudo (vast.ai / runpod / lambda cuda-devel image):
bash scripts/setup_cloud.sh      # apt libsparsehash + pip + torchsparse + smoke test

# data — SemanticKITTI velodyne+labels (~80GB), then point the config at it:
bash scripts/download_semantickitti.sh /data/semantickitti
#   -> edit configs/data/semantickitti.yaml  root: /data/semantickitti/dataset
```

No GPU/data? Verify the pipeline without torchsparse or the dataset:
```bash
PYTHONPATH=. python -m scripts.smoke_test   # synthetic: collate/heads/losses/mIoU
```

## Run
```bash
python -m src.train task=semantic model=minkunet     # GATE 1: reproduce mIoU (seq 08 val)
python -m src.train task=panoptic model=spvcnn data.voxel=0.05   # GATE 2: +center/offset
python -m src.eval  ckpt=runs/best.ckpt task=panoptic           # PQ/mIoU + per-class table + FPS
python -m scripts.viz ckpt=runs/best.ckpt viz.frame=000100 viz.save=demo/   # Open3D PNGs
```

## Upstream to adapt (semantic gate)
- Backbone + SemanticKITTI loader: **mit-han-lab/spvnas** (SPVCNN, torchsparse).
- Panoptic eval reference: **PRBonn/semantic-kitti-api** (`evaluate_panoptic`).

## Layout
```
configs/      Hydra (data / model / train)
src/data/     SemanticKITTI dataset + label maps (panoptic: semantic + instance)
src/models/   backbone wrapper + semantic/offset/center heads
src/panoptic/ offset-shift clustering + PQ eval
src/viz/      Open3D rendering
scripts/      train/eval/infer/seed helpers
docker/       reproducible env
DESIGN.md     technical decisions & tradeoffs   TASKS.md  8-week plan w/ gates
```
