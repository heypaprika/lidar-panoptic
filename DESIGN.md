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
MinkUNet U-Net (spconv)  ──►  per-point features
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
cheap on a single 24 GB GPU (no second network).

## 2.1 Instance targets, losses & grouping (GATE 2)
Notation: point `p` at `x_p ∈ ℝ³`; predicted semantic `ŷ_p`, center `ĥ_p ∈ [0,1]`, offset `ô_p ∈ ℝ³`.
`T` = *thing* points (a point whose **GT** instance id > 0 and whose class is a thing class).

**Targets** (`_instance_targets`). Instances are keyed by **(scan, instance-id)** so the same raw
id in two scans of a batch never merges. For thing point `p` belonging to instance `k`:
```
centroid   c_k  = mean_{q∈k} x_q            # per-instance mean of xyz
offset_gt  o*_p = c_k − x_p                 # points to the instance center
center_gt  h*_p = exp( −‖o*_p‖² / 2σ² )     # Gaussian heatmap, σ = center_sigma (1.0 m)
```
Stuff / ignore points get `o* = 0`, `h* = 0` (center head regresses 0 away from thing centers).

**Losses.** Total (task=panoptic) is semantic + instance:
```
L_sem = λ_ce · CE(ŷ, y; ignore=0) + λ_lov · Lovász-softmax(softmax(ŷ), y; ignore=0)
L_ctr = MSE( ĥ , h* )                       # over ALL points (heatmap regression)
L_off = (1/|T|) · Σ_{p∈T} ‖ ô_p − o*_p ‖₁   # L1, thing points only
L     = L_sem + λ_ctr · L_ctr + λ_off · L_off
```
Offset is masked to `T` (stuff has no center); center is dense so the head learns "thing-ness".
Weights `λ` and `σ` live in `configs/config.yaml (loss.*)`.

**Grouping** (`panoptic_from_offsets`, inference, per scan). Shift each thing point to its predicted
center and cluster **per predicted class**:
```
x'_p = x_p + ô_p                            # offset-shifted coords collapse toward centers
for cls in thing classes:
    idx = { p : ŷ_p = cls }                 # ≥ min_points, else skip
    labels = DBSCAN(eps, min_samples=min_points).fit(x'_{idx})
    each non-noise cluster → a new global instance id; DBSCAN noise (−1) → no instance
```
> Note: this baseline groups from **offset only** — the center head supplies auxiliary "thing-ness"
> supervision and is the hook for the **center-NMS grouping ablation** (seed at heatmap peaks,
> assign by nearest shifted center), not consumed by DBSCAN itself. Honest about what each head does.

**Merge → panoptic label** `(semantic_id, instance_id)` per point:
- **stuff** point → `(ŷ_p, 0)`.
- **thing** point in a cluster → `(ŷ_p, cluster_id)`.
- **thing** point left unclustered (noise / class below `min_points`) → `(ŷ_p, 0)` — semantic only,
  no false instance. The official evaluator then scores it as an unmatched thing region.

## 3. Compute plan (rented cloud GPU, 24 GB+)
Dev is done on a small local box (12 GB, ~7 GB free) — code + an spconv-free **synthetic smoke
test** (`scripts/smoke_test.py`) only. Real training runs on a **rented cloud GPU** (24 GB+ VRAM,
~170 GB disk for velodyne+labels); the local box can't hold the ~80 GB dataset. So we de-risk:
- voxel **0.05 m**, small batch (2–4 scans), **mixed precision (fp16)**, gradient accumulation.
- **Gate 1 first**: reproduce SPVCNN *semantic* mIoU near published before touching instances —
  if this fails, nothing downstream matters.
- Instance heads are lightweight → train jointly after the backbone is healthy (optionally warm
  from the semantic checkpoint and short-train the heads).
- If time/VRAM bites: train/report on a **reduced setting** (subsequence, fewer epochs, lower
  res). Being explicit about the setting in the README costs no credibility; a broken run does.

## 4. Evaluation
Per class `c`, match predicted vs GT segments (same class, **IoU > 0.5** ⇒ unique TP):
```
SQ_c = (Σ_{(p,g)∈TP} IoU(p,g)) / |TP|                    # avg IoU of matched segments
RQ_c = |TP| / ( |TP| + ½|FP| + ½|FN| )                   # detection F1
PQ_c = SQ_c · RQ_c ;   PQ = mean_c PQ_c
PQ†  = ( Σ_{c∈thing} PQ_c + Σ_{c∈stuff} IoU_c ) / (|thing|+|stuff|)   # stuff scored by IoU
```
- Use SemanticKITTI's **official** evaluator (PRBonn `semantic-kitti-api` → `PanopticEval`) — do
  **not** reinvent the >0.5 matching + void handling. `panoptic/pq.py` is a thin adapter
  (accumulate per scan → `getPQ` / `getSemIoU`); it computes PQ† from the per-class PQ/IoU vectors.
- **mIoU** for semantic (standard 19-class, ignore=0), from the same evaluator or `IoUMeter`.
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
| sparse-conv install/version pain | use **spconv** prebuilt CUDA wheel (no source build); Docker image. (Dropped torchsparse: its 2.0.0b strided conv mis-generated coords.) |
| Semantic repro doesn't match | Gate 1 checkpoint; adapt spvnas hyperparams before adding heads |
| PQ implementation bugs | wrap official evaluator, unit-test on toy scene |
| Clustering unstable | start DBSCAN on shifted coords; add dynamic-shift only if needed |
| GPU OOM | fp16 + grad-accum + smaller voxel/batch + reduced setting |
| dataset won't fit local box | dev = synthetic smoke test; train on cloud GPU w/ ~170 GB disk |

## 8. Open decisions (revisit as we build)
- Backbone depth (SPVCNN cr=1.0 vs 0.5 for speed/VRAM).
- Clustering: DBSCAN vs learned dynamic-shift (DS-Net) — ablate.
- Center head use: auxiliary supervision only vs center-NMS seed grouping (§2.1 note) — ablate.
- `center_sigma` (heatmap width) and DBSCAN `eps`/`min_points` — tune on val.
- Instance grouping over full scan vs tiles (memory).

## 9. Extension: temporal / multi-view consistency (where this goes next)
This repo does **single-scan** panoptic. The natural next step — and the one that matters for
spatial-intelligence / HD-mapping perception — is making instance identities **consistent across
frames**, not re-segmented independently each scan.

- **4D panoptic (LiDAR, temporal):** SemanticKITTI already defines a *4D panoptic* task — the same
  instance must keep its id across consecutive scans. Our center+offset design extends cleanly:
  register successive scans into a common frame (ego-motion / poses ship with KITTI), associate
  instances across time by centroid proximity + offset flow, and add a light tracking/association
  head. Metric moves from PQ to **LSTQ** (association + segmentation quality).
- **Multi-view consistency (images):** the same principle in the multi-view image setting —
  segment once, keep instances coherent across viewpoints via the shared 3D structure — is exactly
  the direction of recent work like NAVER LABS Europe's *PanSt3R* (multi-view-consistent panoptic
  on a 3D reconstruction backbone). The bridge from here: our per-point instance embedding/offset is
  the LiDAR analogue of enforcing cross-view instance agreement through reconstructed geometry.
- **Why the current design transfers:** bottom-up center/offset is geometry-native (operates on
  3D coordinates), so lifting it from one scan to a fused multi-scan / multi-view point set is an
  association problem on top of the same heads — not a new architecture. That is the intended growth
  path, and the reason the instance branch is kept modular.
