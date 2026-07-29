# Ablations

Pre-registered experiments — hypotheses and setups written **before** the runs, results filled in
after. Each isolates one variable against the same GATE-2 baseline (SPVCNN, voxel 0.05 m, center +
offset, offset-shift DBSCAN), val = seq 08, official evaluator. Scope: run **≥1** end-to-end;
the rest can ship as designed-but-not-run with the hypothesis stated. See `DESIGN.md` for the math.

**Baseline (reference for every Δ below)**

| mIoU | PQ | PQ† | SQ | RQ | FPS |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

Fill from `python -m src.eval ckpt=runs/best.ckpt task=panoptic`. Δ columns below are `variant − baseline`.

---

## A1 · Instance targets: center+offset vs embedding+MeanShift
**Question:** does bottom-up offset regression beat metric-learning embeddings for stable, non-trivial PQ?
**Hypothesis:** offset regression reaches useful PQ faster and more stably; embeddings are sensitive
to margin/bandwidth (DESIGN §2).
**Setup:** swap the offset/center heads for an instance-embedding head + discriminative loss; group
with MeanShift instead of DBSCAN. Backbone, voxel, epochs, semantic loss held fixed.
**Primary metric:** PQ (+ training stability: epochs-to-target, variance across seeds).

| Variant | PQ | ΔPQ | SQ | RQ | notes (convergence/stability) |
|---|---|---|---|---|---|
| center+offset (baseline) | — | 0 | — | — | — |
| embedding + MeanShift | — | — | — | — | — |

**Takeaway:** _fill after run._

---

## A2 · Clustering: DBSCAN vs learned dynamic-shift (DS-Net)
**Question:** is a learned dynamic point-shift worth the complexity over a fixed DBSCAN on shifted coords?
**Hypothesis:** dynamic-shift helps crowded scenes (adjacent instances) at some FPS cost.
**Setup:** keep the same heads; replace `panoptic_from_offsets`'s DBSCAN with iterative dynamic-shift
grouping. Same eps-equivalent bandwidth swept on val.
**Primary metric:** PQ (esp. on thing-dense classes: car/person/bicyclist) + FPS.

| Variant | PQ | ΔPQ | thing-PQ | FPS |
|---|---|---|---|---|
| DBSCAN (baseline) | — | 0 | — | — |
| dynamic-shift | — | — | — | — |

**Takeaway:** _fill after run._

---

## A3 · Center head: auxiliary supervision vs center-NMS grouping
**Question:** does actually *using* the center heatmap for grouping beat treating it as aux supervision only?
**Hypothesis:** center-NMS seeding reduces over-segmentation vs pure-offset DBSCAN on large instances.
**Setup:** baseline trains the center head but groups from offset only. Variant seeds instances at
heatmap peaks (NMS) and assigns thing points by nearest shifted center. Same weights/σ.
**Primary metric:** PQ + RQ (fragmentation), qualitative viz on trucks/other-vehicle.

| Variant | PQ | ΔPQ | RQ | over-seg (inst/scan) |
|---|---|---|---|---|
| offset-only DBSCAN (baseline) | — | 0 | — | — |
| center-NMS grouping | — | — | — | — |

**Takeaway:** _fill after run._

---

## A4 · Backbone width: SPVCNN cr = 1.0 vs 0.5
**Question:** the PQ↔FPS tradeoff of halving channel width.
**Hypothesis:** cr=0.5 loses a few PQ points for a large FPS/VRAM win — useful for deployment framing.
**Setup:** `model=spvcnn model.cr=0.5` vs `1.0`; everything else fixed.
**Primary metric:** PQ vs FPS (and peak VRAM).

| cr | mIoU | PQ | FPS | VRAM |
|---|---|---|---|---|
| 1.0 (baseline) | — | — | — | — |
| 0.5 | — | — | — | — |

**Takeaway:** _fill after run._

---

## A5 · Voxel size: 0.05 m vs 0.10 m
**Question:** resolution vs speed/memory.
**Hypothesis:** 0.10 m roughly doubles throughput and cuts VRAM, costing PQ mostly on small things
(bicycle/pole).
**Setup:** `data.voxel=0.05` vs `0.10`; same epochs.
**Primary metric:** PQ (overall + small-thing classes) vs FPS/VRAM.

| voxel | mIoU | PQ | small-thing PQ | FPS | VRAM |
|---|---|---|---|---|---|
| 0.05 m (baseline) | — | — | — | — | — |
| 0.10 m | — | — | — | — | — |

**Takeaway:** _fill after run._
