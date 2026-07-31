# DESIGN — 희소 포인트클라우드 Panoptic Segmentation

이 문서는 **결정과 트레이드오프**를 기록합니다 — "만들 줄 아는 사람"과 "돌려만 본 사람"을 가르는, 리뷰어가
실제로 읽는 부분입니다.

## 1. 문제 정의
SemanticKITTI panoptic = 점별 **semantic class**(19 + ignore) **+ instance id**(8개 *thing* 클래스:
car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist). *stuff* 클래스
(road, building, …)는 instance가 없습니다. 지표: **PQ**(= SQ × RQ), **PQ†**, 그리고 semantic에 대한
**mIoU**.

## 2. 아키텍처 (bottom-up, 단일 백본)
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
**center+offset(Panoptic-DeepLab / DS-Net)을 embedding+MeanShift보다 택한 이유:** 회귀 타깃이 조밀하고
잘 정의돼 있어 학습이 안정적이고 쓸 만한 PQ에 빨리 도달합니다. metric-learning embedding + MeanShift는
margin/bandwidth에 민감하고 수렴이 느려 **ablation**으로만 남겨 대안을 이해하고 있음을 보입니다.

**백본 하나에 헤드 둘인 이유:** "semantic 모델을 panoptic으로 확장한다"는 서사에 맞고, 단일 24 GB GPU에서
값싸게 돌아갑니다(두 번째 네트워크가 없음).

## 2.1 Instance 타깃·손실·그룹핑 (GATE 2)
표기: 점 `p`의 좌표 `x_p ∈ ℝ³`, 예측 semantic `ŷ_p`, center `ĥ_p ∈ [0,1]`, offset `ô_p ∈ ℝ³`.
`T` = *thing* 점(= **GT** instance id > 0 이고 클래스가 thing인 점).

**타깃**(`_instance_targets`). instance는 **(scan, instance-id)**로 키를 잡아, 배치 내 두 스캔의 같은 raw id가
섞이지 않게 합니다. instance `k`에 속한 thing 점 `p`에 대해:
```
centroid   c_k  = mean_{q∈k} x_q            # per-instance mean of xyz
offset_gt  o*_p = c_k − x_p                 # points to the instance center
center_gt  h*_p = exp( −‖o*_p‖² / 2σ² )     # Gaussian heatmap, σ = center_sigma (1.0 m)
```
stuff / ignore 점은 `o* = 0`, `h* = 0`(center 헤드는 thing 중심에서 멀어질수록 0으로 회귀).

**손실.** 전체(task=panoptic)는 semantic + instance:
```
L_sem = λ_ce · CE(ŷ, y; ignore=0) + λ_lov · Lovász-softmax(softmax(ŷ), y; ignore=0)
L_ctr = MSE( ĥ , h* )                       # over ALL points (heatmap regression)
L_off = (1/|T|) · Σ_{p∈T} ‖ ô_p − o*_p ‖₁   # L1, thing points only
L     = L_sem + λ_ctr · L_ctr + λ_off · L_off
```
offset은 `T`로 마스킹하고(stuff는 중심이 없음), center는 전체 점에 대해 조밀하게 걸어 헤드가 "thing-ness"를
학습하게 합니다. 가중치 `λ`와 `σ`는 `configs/config.yaml (loss.*)`에 있습니다.

**그룹핑**(`panoptic_from_offsets`, 추론, 스캔별). 각 thing 점을 예측 중심으로 이동시킨 뒤 **예측 클래스별로**
clustering:
```
x'_p = x_p + ô_p                            # offset-shifted coords collapse toward centers
for cls in thing classes:
    idx = { p : ŷ_p = cls }                 # ≥ min_points, else skip
    labels = DBSCAN(eps, min_samples=min_points).fit(x'_{idx})
    each non-noise cluster → a new global instance id; DBSCAN noise (−1) → no instance
```
> 참고: 이 baseline은 **offset만으로** 그룹핑합니다 — center 헤드는 보조 "thing-ness" supervision을 주고,
> **center-NMS 그룹핑 ablation**(heatmap peak를 seed로, 가장 가까운 shifted center에 할당)의 훅입니다.
> DBSCAN 자체는 center를 쓰지 않습니다. 각 헤드가 실제로 하는 일을 정직하게 구분합니다.

**Merge → panoptic label** `(semantic_id, instance_id)`, 점별:
- **stuff** 점 → `(ŷ_p, 0)`.
- 클러스터에 속한 **thing** 점 → `(ŷ_p, cluster_id)`.
- 클러스터에 못 든 **thing** 점(noise / `min_points` 미만 클래스) → `(ŷ_p, 0)` — semantic만, 거짓 instance
  없음. 공식 evaluator가 이를 매칭 안 된 thing 영역으로 채점합니다.

## 3. 컴퓨트 계획 (대여 클라우드 GPU, 24 GB+)
개발은 작은 로컬 박스(12 GB, 여유 ~7 GB)에서 코드 + **합성 스모크 테스트**(`scripts/smoke_test.py`,
spconv 불필요)만 합니다. 실제 학습은 **대여 클라우드 GPU**(24 GB+ VRAM, velodyne+labels용 ~170 GB 디스크)
에서 합니다 — 로컬은 ~80 GB 데이터셋을 담지 못합니다. 그래서 리스크를 낮춥니다:
- voxel **0.05 m**, 작은 batch(2–4 스캔), **fp32**(spconv Native + AMP 조합이 불안정 — 엔지니어링 노트
  참고), gradient accumulation.
- **Gate 1 먼저**: instance를 건드리기 전에 semantic mIoU를 공개 수치 근처로 재현 — 여기서 실패하면 이후는
  의미가 없다.
- instance 헤드는 가벼워, 백본이 건강해진 뒤 함께 학습(옵션: semantic 체크포인트에서 warm-start 후 헤드만
  단기 학습).
- 시간/VRAM이 부족하면 **축소 설정**(subsequence, 적은 epoch, 낮은 해상도)으로 학습·보고. README에 설정을
  명시하는 건 신뢰를 깎지 않지만, 깨진 런은 깎는다.

## 4. 평가
클래스 `c`별로 예측·GT 세그먼트를 매칭(같은 클래스, **IoU > 0.5** ⇒ 유일 TP):
```
SQ_c = (Σ_{(p,g)∈TP} IoU(p,g)) / |TP|                    # avg IoU of matched segments
RQ_c = |TP| / ( |TP| + ½|FP| + ½|FN| )                   # detection F1
PQ_c = SQ_c · RQ_c ;   PQ = mean_c PQ_c
PQ†  = ( Σ_{c∈thing} PQ_c + Σ_{c∈stuff} IoU_c ) / (|thing|+|stuff|)   # stuff scored by IoU
```
- SemanticKITTI **공식** evaluator(PRBonn `semantic-kitti-api` → `PanopticEval`)를 사용 — >0.5 매칭과
  void 처리를 **직접 재구현하지 않는다**. `panoptic/pq.py`는 얇은 어댑터(스캔별 누적 → `getPQ` / `getSemIoU`)
  이고, per-class PQ/IoU 벡터로 PQ†를 계산한다.
- semantic **mIoU**(표준 19-class, ignore=0), 같은 evaluator 또는 `IoUMeter`로.
- **FPS / latency** 보고(단일 GPU 추론) — panoptic은 돌아가야 쓸모가 있다.

## 5. 리서치 엔지니어링 (차별점)
Hydra config · PyTorch Lightning · Weights & Biases · Docker · Open3D 시각화 · `ablations.md`.
이것들은 JD의 *"Training Data Pipeline / infra"* 언어에 바로 대응하며, 레포를 한눈에 읽히게 합니다.

## 6. 범위 & 비범위
- **핵심(반드시 완료):** semantic 재현 → center/offset panoptic → PQ → viz → ablation 1개.
- **스트레치(빼도 핵심 무손상):** CARLA 합성 LiDAR → 소규모 **sim-to-real**(합성 pretrain 후 SemanticKITTI
  fine-tune, 또는 self-training). 시간이 부족하면 **설계된 실험 + 예비 수치**로 제출하지, 반쯤 깨진 기능으로
  내지 않는다.
- **명시적 제외:** sparse 백본을 밑바닥부터 구현(spvnas 적응으로 대체), 프로덕션 OLAP, 손쉬운 DDP를 넘어선
  멀티-GPU 분산 학습.

## 7. 리스크 → 완화
| 리스크 | 완화 |
|---|---|
| sparse-conv 설치/버전 문제 | **spconv** 프리빌트 CUDA 휠 사용(소스 빌드 없음), Docker 이미지. (torchsparse 폐기: 2.0.0b strided conv가 좌표를 잘못 생성.) |
| batch>2에서 illegal memory access | spconv의 int32 좌표 flatten 오버플로 — 점군을 고정 범위로 crop해 볼륨 bound(엔지니어링 노트 §3). |
| semantic 재현이 안 맞음 | Gate 1 체크포인트; 헤드 붙이기 전 하이퍼파라미터 조정 |
| PQ 구현 버그 | 공식 evaluator를 감싸고, toy 장면으로 단위 검증 |
| clustering 불안정 | shifted 좌표에 DBSCAN부터; 필요할 때만 dynamic-shift 추가 |
| GPU OOM | fp32 + grad-accum + 작은 voxel/batch + 축소 설정 |
| 데이터셋이 로컬에 안 들어감 | 개발 = 합성 스모크 테스트; ~170 GB 디스크 클라우드 GPU에서 학습 |

## 8. 열린 결정 (만들며 재검토)
- 백본 폭(SPVCNN cr=1.0 vs 0.5, 속도/VRAM).
- clustering: DBSCAN vs 학습형 dynamic-shift(DS-Net) — ablation.
- center 헤드 용도: 보조 supervision만 vs center-NMS seed 그룹핑(§2.1 참고) — ablation.
- `center_sigma`(heatmap 폭)와 DBSCAN `eps`/`min_points` — val에서 튜닝.
- instance 그룹핑을 전체 스캔 vs 타일 단위로(메모리).

## 9. 확장: 시간축 / 멀티뷰 일관성 (다음 방향)
이 레포는 **단일 스캔** panoptic입니다. 자연스러운 다음 단계 — 공간지능 / HD-매핑 인지에서 중요한 지점 — 은
매 스캔 독립적으로 다시 세그멘테이션하지 않고 instance id를 **프레임 간에 일관**되게 유지하는 것입니다.

- **4D panoptic (LiDAR, 시간축):** SemanticKITTI에는 이미 *4D panoptic* 태스크가 있습니다 — 같은 instance가
  연속 스캔에서 id를 유지해야 합니다. 우리의 center+offset 설계는 깔끔하게 확장됩니다: 연속 스캔을 공통 좌표계로
  정합(ego-motion / pose는 KITTI에 포함), centroid 근접성 + offset flow로 시간축 association, 가벼운
  tracking/association 헤드 추가. 지표는 PQ에서 **LSTQ**(association + segmentation quality)로 이동.
- **멀티뷰 일관성 (이미지):** 멀티뷰 이미지 세팅에서의 같은 원리 — 한 번 세그멘테이션하고 공유 3D 구조를 통해
  뷰 간 instance를 일관되게 유지 — 는 NAVER LABS Europe의 *PanSt3R*(3D 재구성 백본 위의 멀티뷰 일관 panoptic)
  같은 최근 연구의 방향과 정확히 같습니다. 여기서의 다리: 우리의 점별 instance embedding/offset은, 재구성된
  기하를 통해 뷰 간 instance 합의를 강제하는 것의 LiDAR 대응물입니다.
- **왜 현재 설계가 전이되나:** bottom-up center/offset은 geometry-native(3D 좌표에서 동작)라, 이를 융합된
  멀티스캔 / 멀티뷰 점집합으로 올리는 것은 새 아키텍처가 아니라 **같은 헤드 위의 association 문제**입니다. 이것이
  의도한 성장 경로이며, instance 브랜치를 모듈식으로 유지한 이유입니다.
