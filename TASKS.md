# TASKS — 8-week plan (with go/no-go gates)

Core = weeks 1–5. Stretch (sim-to-real) = weeks 6–7. Polish = week 8.
A gate that fails **blocks** the next phase — fix it before moving on.

## Week 1–2 · Semantic backbone repro  ⛔ GATE 1
- [ ] SemanticKITTI dataset + label map wired (`src/data/`), sanity-check a scan in Open3D.
- [ ] SPVCNN backbone via torchsparse (adapt **spvnas**), semantic head only.
- [ ] Train + eval **mIoU** on val (seq 08).
- **GATE 1:** semantic mIoU is *close to published* (reduced setting ok). If not → stop & fix.

## Week 3–4 · Panoptic heads  ⛔ GATE 2
- [ ] Center head (centerness heatmap) + Offset head (3D offset to instance center).
- [ ] Losses: CE + Lovász (sem), MSE (center), L1 (offset, thing points).
- [ ] Offset-shift + DBSCAN clustering → instance ids → merge to panoptic.
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
