# TASKS — 8-week plan (with go/no-go gates)

Core = weeks 1–5. Stretch (sim-to-real) = weeks 6–7. Polish = week 8.
A gate that fails **blocks** the next phase — fix it before moving on.

## Week 1–2 · Semantic backbone repro  ⛔ GATE 1
- [x] SemanticKITTI dataset + label map (`src/data/`) — remap/instance/mIoU **unit-verified**.
- [x] Voxelize + collate (numpy, version-robust) + DataModule.
- [x] Semantic training wired: CE + Lovász, val **mIoU**, Hydra/Lightning entrypoint.
- [x] Runnable backbone: **MinkUNet (torchsparse)** — `python -m src.train task=semantic`.
- [x] **Pipeline smoke test** (`scripts/smoke_test.py`, dummy backbone, synthetic pts) — collate /
      heads / Lovász / CE / IoU verified end-to-end on torch 2.4 + CUDA. torchsparse-independent.
- [ ] **Verify torchsparse v2.1 API** on a box with the sparse-conv build (marked `# VERIFY` in
      `backbone.py`) — this dev box has no sudo for libsparsehash + too little disk for the dataset.
- [ ] Run: reproduce semantic **mIoU** on val (seq 08). Sanity-check a scan in Open3D.
- [ ] (upgrade) swap MinkUNet → **SPVCNN** (vendor spvnas) for the headline number.
- **GATE 1:** semantic mIoU *close to published* (reduced setting ok). If not → stop & fix.

## Week 3–4 · Panoptic heads  ⛔ GATE 2
- [x] Center head (centerness heatmap) + Offset head (3D offset to instance center).
- [x] Instance targets + losses: per-(scan,inst) centroid → offset_gt/center_gt; MSE(center),
      L1(offset, thing points). `_instance_targets`/`_instance_loss` **smoke-verified** (grads flow).
- [x] Offset-shift + DBSCAN clustering → instance ids, wired into val (`_accumulate_panoptic`).
- [x] PQ adapter (`panoptic/pq.py`) over official PanopticEval; PQ/PQ†/SQ/RQ/mIoU + lazy vendor.
- [ ] **Vendor** `np_ioueval.py` (PRBonn) + run: **non-trivial PQ** on val seq 08 (needs cloud box).
- **GATE 2:** non-trivial **PQ** on val (official evaluator). If not → debug heads/clustering.

## Week 5 · Eval + viz
- [ ] Official PQ / PQ† / SQ / RQ wrapper; per-class table.
- [ ] Open3D renderer: semantic vs panoptic (instance-colored) side by side; demo frames.
- [ ] Latency/FPS measurement.

## Week 6–7 · Sim-to-Real (STRETCH — droppable)
- [ ] CARLA synthetic LiDAR + auto label export → point-cloud dataset.
- [ ] Small experiment: pretrain synthetic → fine-tune SemanticKITTI (or self-training).
- [ ] If blocked: ship as **designed experiment + preliminary numbers** in README.

## Week 8 · Ship
- [ ] Results tables (mIoU/PQ + ablation: center-offset vs embedding+MeanShift).
- [ ] README polish, architecture figure, **demo video/gif**, short tech blog.
- [ ] Dockerfile reproduces train/eval.

## Ablations to run (pick ≥1)
- center-offset **vs** instance-embedding+MeanShift.
- clustering: DBSCAN **vs** dynamic-shift (DS-Net).
- backbone width (SPVCNN cr 1.0 vs 0.5): PQ vs FPS tradeoff.
- voxel size 0.05 vs 0.10 m.
