# Ablations

사전 등록 실험 — 가설과 setup을 런 **이전에** 적고, 결과는 이후에 채웁니다. 각 실험은 동일한 GATE-2 baseline
(spconv MinkUNet, voxel 0.05 m, center + offset, offset-shift DBSCAN, val = seq 08, 공식 evaluator)에
대해 변수 하나만 분리합니다. 범위: **≥1개**를 end-to-end로 실행하고, 나머지는 가설을 명시한 "설계했으나
미실행"으로 남깁니다. 수식은 `DESIGN.md` 참고.

**Baseline (아래 모든 Δ의 기준)**

| mIoU | PQ | PQ† | SQ | RQ | FPS |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

`python -m src.eval ckpt=<best.ckpt> task=panoptic`로 채웁니다. 아래 Δ 열은 `variant − baseline`.

---

## A1 · Instance 타깃: center+offset vs embedding+MeanShift
**질문:** bottom-up offset 회귀가 안정적이고 non-trivial한 PQ에서 metric-learning embedding을 이기는가?
**가설:** offset 회귀가 쓸 만한 PQ에 더 빨리·안정적으로 도달; embedding은 margin/bandwidth에 민감(DESIGN §2).
**Setup:** offset/center 헤드를 instance-embedding 헤드 + discriminative loss로 교체; DBSCAN 대신
MeanShift로 그룹핑. 백본·voxel·epoch·semantic loss는 고정.
**주 지표:** PQ (+ 학습 안정성: 목표까지 epoch 수, seed 간 분산).

| Variant | PQ | ΔPQ | SQ | RQ | 비고(수렴/안정성) |
|---|---|---|---|---|---|
| center+offset (baseline) | — | 0 | — | — | — |
| embedding + MeanShift | — | — | — | — | — |

**결론:** _런 이후 기입._

---

## A2 · Clustering: DBSCAN vs 학습형 dynamic-shift (DS-Net)
**질문:** 학습형 dynamic point-shift가 shifted 좌표 위 고정 DBSCAN 대비 복잡도를 감수할 가치가 있나?
**가설:** dynamic-shift는 밀집 장면(인접 instance)에서 도움되나 FPS 비용이 있다.
**Setup:** 헤드는 그대로 두고, `panoptic_from_offsets`의 DBSCAN을 반복 dynamic-shift 그룹핑으로 교체.
val에서 동등한 bandwidth를 스윕.
**주 지표:** PQ (특히 thing-밀집 클래스: car/person/bicyclist) + FPS.

| Variant | PQ | ΔPQ | thing-PQ | FPS |
|---|---|---|---|---|
| DBSCAN (baseline) | — | 0 | — | — |
| dynamic-shift | — | — | — | — |

**결론:** _런 이후 기입._

---

## A3 · Center 헤드: 보조 supervision vs center-NMS 그룹핑
**질문:** center heatmap을 실제로 그룹핑에 *쓰는* 것이 보조 supervision으로만 두는 것보다 나은가?
**가설:** center-NMS seeding이 큰 instance에서 pure-offset DBSCAN 대비 over-segmentation을 줄인다.
**Setup:** baseline은 center 헤드를 학습하되 offset만으로 그룹핑. Variant는 heatmap peak를 seed(NMS)로,
thing 점을 가장 가까운 shifted center에 할당. 가중치/σ 동일.
**주 지표:** PQ + RQ(분절), truck/other-vehicle 정성 viz.

| Variant | PQ | ΔPQ | RQ | over-seg (inst/scan) |
|---|---|---|---|---|
| offset-only DBSCAN (baseline) | — | 0 | — | — |
| center-NMS 그룹핑 | — | — | — | — |

**결론:** _런 이후 기입._

---

## A4 · 백본 폭: cr = 1.0 vs 0.5
**질문:** 채널 폭 절반의 PQ↔FPS 트레이드오프.
**가설:** cr=0.5는 PQ 몇 점을 내주고 FPS/VRAM 이득이 크다 — 배포 관점의 트레이드오프.
**Setup:** `model.cr=0.5` vs `1.0`; 나머지 고정.
**주 지표:** PQ vs FPS (및 peak VRAM).

| cr | mIoU | PQ | FPS | VRAM |
|---|---|---|---|---|
| 1.0 (baseline) | — | — | — | — |
| 0.5 | — | — | — | — |

**결론:** _런 이후 기입._

---

## A5 · Voxel 크기: 0.05 m vs 0.10 m
**질문:** 해상도 vs 속도/메모리.
**가설:** 0.10 m는 처리량을 대략 2배로 늘리고 VRAM을 줄이며, PQ 손실은 주로 작은 thing(bicycle/pole)에서.
**Setup:** `data.voxel=0.05` vs `0.10`; epoch 동일.
**주 지표:** PQ (전체 + small-thing 클래스) vs FPS/VRAM.

| voxel | mIoU | PQ | small-thing PQ | FPS | VRAM |
|---|---|---|---|---|---|
| 0.05 m (baseline) | — | — | — | — | — |
| 0.10 m | — | — | — | — | — |

**결론:** _런 이후 기입._
