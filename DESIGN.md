# DESIGN — Lightweight Panoptic Segmentation on Sparse Point Clouds

Target JD signal: 3D perception (NAVER-LABS-style). This doc records the **decisions and
tradeoffs** — the part interviewers read to tell a builder from a runner.

## 1. Problem framing
SemanticKITTI panoptic = per-point **semantic class** (19 + ignore) **+ instance id** for the 8
*thing* classes (car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist,
motorcyclist). *Stuff* classes (road, building, …) have no instances. Metric: **PQ** (= SQ × RQ),
plus PQ†, and **mIoU** for the semantic part.

## 2. Architecture (bottom-up, single backbone)
```
points (x,y,z,remission)
      │  voxelize (0.05 m)
      ▼
SPVCNN backbone (torchsparse)  ──►  per-point features
      ├── Semantic head   → class logits (19+1)            [CE + Lovász-softmax]
      ├── Center head     → centerness heatmap (thing)     [MSE]
      └── Offset head     → 3D offset to instance center   [L1, thing points only]
      ▼
shift thing points by predicted offset → cluster (DBSCAN / dynamic-shift)
      ▼
merge: stuff = semantic; thing = (semantic argmax) × (cluster id)  → PANOPTIC
```
**Why center+offset (Panoptic-DeepLab / DS-Net) over embedding+MeanShift:** regression targets
are dense and well-posed → trains stably and reaches useful PQ quickly. Metric-learning
embeddings + MeanShift are sensitive to margin/bandwidth and slow to converge → kept only as an
**ablation** to show we understand the alternative.

**Why one backbone, two heads:** matches the "extend a semantic model to panoptic" story and is
cheap on a 24 GB GPU (no second network).

## 3. Compute plan (local 24 GB, 3090/4090)
Full SemanticKITTI is heavy, so we de-risk:
- voxel **0.05 m**, small batch (2–4 scans), **mixed precision (fp16)**, gradient accumulation.
- **Gate 1 first**: reproduce SPVCNN *semantic* mIoU near published before touching instances —
  if this fails, nothing downstream matters.
- Instance heads are lightweight → train jointly after the backbone is healthy (optionally warm
  from the semantic checkpoint and short-train the heads).
- If time/VRAM bites: train/report on a **reduced setting** (subsequence, fewer epochs, lower
  res). Being explicit about the setting in the README costs no credibility; a broken run does.

## 4. Evaluation
- **PQ / PQ† / SQ / RQ**: use SemanticKITTI's official panoptic evaluator (PRBonn
  `semantic-kitti-api`) — do **not** reinvent (it has subtle IoU-matching + void handling).
- **mIoU** for semantic (standard 19-class, ignore=0).
- Report **FPS / latency** (inference on one GPU) — panoptic is only useful if it runs.

## 5. Research engineering (the differentiator)
Hydra configs · PyTorch Lightning · Weights & Biases · Docker · Open3D visualization ·
`ablations.md`. These map directly to the JD's *"Training Data Pipeline / infra"* language and
make the repo legible at a glance.

## 6. Scope & anti-scope
- **Core (must finish):** SPVCNN semantic repro → center/offset panoptic → PQ → viz → 1 ablation.
- **Stretch (drop without hurting core):** CARLA synthetic LiDAR → small **sim-to-real**
  (pretrain synthetic, fine-tune SemanticKITTI, or self-training). If time runs out, ship it as a
  **designed experiment + preliminary numbers**, not a half-broken feature.
- **Explicitly out:** building the sparse backbone from scratch (adapt spvnas), production OLAP,
  multi-GPU distributed training beyond DDP-if-easy.

## 7. Risks → mitigations
| Risk | Mitigation |
|---|---|
| torchsparse/CUDA install pain | pin torchsparse v2.1, `libsparsehash-dev`, Docker image |
| Semantic repro doesn't match | Gate 1 checkpoint; adapt spvnas hyperparams before adding heads |
| PQ implementation bugs | wrap official evaluator, unit-test on toy scene |
| Clustering unstable | start DBSCAN on shifted coords; add dynamic-shift only if needed |
| 24 GB OOM | fp16 + grad-accum + smaller voxel/batch + reduced setting |

## 8. Open decisions (revisit as we build)
- Backbone depth (SPVCNN cr=1.0 vs 0.5 for speed/VRAM).
- Clustering: DBSCAN vs learned dynamic-shift (DS-Net) — ablate.
- Instance grouping over full scan vs tiles (memory).
