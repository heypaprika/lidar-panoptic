# Failure analysis

panoptic 예측의 실패 모드를 정성·정량으로 정리한다. 각 항목은 **관찰 → 원인 가설 → 확인 방법 → (그림)**
구조다. 그림은 학습된 체크포인트로 `scripts/viz`(및 error map)를 뽑아 채운다.

> 방법: `python -m scripts.viz ckpt=<best.ckpt> viz.frame=<f> viz.save=demo/ data.root=$DATA` 로
> semantic / instance / panoptic 렌더를 만들고, GT와 대비해 오분류·오분할 프레임을 고른다.

## 정량 — 어디서 가장 실패하나
semantic(mIoU 54.8) per-class IoU에서 최악 클래스. 흔한 클래스는 포화(car 94.3, road 89.7, building
86.9)이므로 격차는 아래 소수 클래스에 집중된다. (panoptic PQ 열은 GATE 2 eval 후 기입.)

| 클래스 | IoU | (PQ) | 원인 가설 |
|---|---|---|---|
| motorcyclist | 0.0 | _tbd_ | 극희귀 thing — augmentation/oversampling 없이 학습 안 됨 |
| other-ground | 0.3 | — | 극희귀 stuff — 정의 모호 + 점 적음 |
| bicycle | 13.7 | _tbd_ | 얇고 점 적음 → semantic·instance 모두 취약 |
| parking | 23.0 | — | road/sidewalk와 경계 모호 |
| other-vehicle | 33.2 | _tbd_ | 클래스 내 형태 다양 + 드묾 |

## 정성 — 대표 실패 모드 (가설)

### F1. 인접 instance가 하나로 merge
- **관찰(예정):** 가까이 붙은 두 사람/차가 한 instance로 묶임.
- **원인 가설:** offset이 두 중심 사이로 평균화되어 shifted 점들이 한 덩어리로 모이고, DBSCAN이 하나로 봄.
- **확인:** 해당 프레임에서 offset-shift 후 점 분포를 보고, `cluster.eps`를 줄였을 때 분리되는지(A2/eps sweep).
- <!-- ![F1](demo/fail_merge.png) GT | Pred | error -->

### F2. bicycle / bicyclist의 불안정한 offset
- **관찰(예정):** 자전거류 instance가 조각나거나(over-seg) 배경으로 샘.
- **원인 가설:** 얇고 점이 적어 중심 추정이 어렵고 offset 회귀 분산이 큼.
- **확인:** thing 클래스별 offset L1 오차 비교, RQ가 특히 낮은지.
- <!-- ![F2](demo/fail_bicycle.png) -->

### F3. 작은 traffic-sign / pole의 약한 center heatmap
- **관찰(예정):** 작은 물체의 center 신호가 약해 검출 누락.
- **원인 가설:** 점 수가 적어 Gaussian center_gt의 유효 신호가 낮고, MSE가 배경 0에 눌림.
- **확인:** center head 출력 heatmap을 시각화, `center_sigma`나 loss 가중치 변화의 영향(ablation).
- <!-- ![F3](demo/fail_sign.png) -->

### F4. 원거리·저밀도 영역
- **관찰(예정):** 센서에서 먼 영역에서 semantic·instance 모두 저하.
- **원인 가설:** 점 밀도가 낮아 voxel feature가 빈약(0.05 m에서 특히), 범위 crop 경계 근처.
- **확인:** 거리 구간별 IoU/PQ, voxel 0.05 vs 0.10 비교(A5).
- <!-- ![F4](demo/fail_range.png) -->

## 정리 (측정 후)
실패 모드가 주로 **작은 thing + 저밀도**에 몰리는지, 그것이 clustering(eps) 탓인지 backbone feature 탓인지를
ablation과 연결해 결론짓는다. (해석은 README Discussion과 연결.)
