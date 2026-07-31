# 희소 포인트클라우드 기반 Panoptic Segmentation (SemanticKITTI)

> 이미지·포인트클라우드 양쪽에서 다뤄온 visual perception 경험을 **완결된 3D LiDAR panoptic 시스템**으로
> 통합한 프로젝트다. 논문 재현을 넘어, semantic segmentation·instance grouping·sparse convolution·대규모
> 평가가 하나의 end-to-end perception 파이프라인으로 맞물리는 과정을 직접 구현하고 검증했다. 특정 modality에
> 갇힌 전문성이 아니라, **연구 아이디어를 작동하는 시스템으로 옮기는 확장 능력**을 보이는 데 목적이 있다.

sparse-voxel semantic 백본(spconv MinkUNet U-Net)에 **center + offset** instance 헤드를 더해 LiDAR
panoptic segmentation을 수행한다. 공식 **PQ / PQ† / SQ / RQ**와 **mIoU**로 평가한다.

<!-- DEMO: scripts.viz가 demo/를 만들면 여기에 결과 이미지:
![semantic vs panoptic](demo/08_000100_panoptic.png)
-->

## Abstract
SemanticKITTI panoptic은 점별 semantic class(19+ignore)와 8개 *thing* 클래스의 instance id를 함께
예측하는 문제다. 본 구현은 semantic 백본을 먼저 재현한 뒤, 같은 백본에 **점별 instance 중심으로의 3D
offset**과 **center heatmap**을 예측하는 두 헤드를 얹는다. 추론 시 thing 점을 예측 offset만큼 이동시켜
중심으로 모은 뒤 클래스별 DBSCAN으로 instance를 만든다(bottom-up, Panoptic-DeepLab / DS-Net 계열).
center+offset을 택한 이유는 희소 포인트에서 embedding+MeanShift보다 회귀 타깃이 조밀·안정적이기
때문이며, 임베딩 방식은 ablation(A1)으로 직접 비교한다. PQ는 직접 구현하지 않고 공식 evaluator를 감싼다.

## Method
```
points (x,y,z,remission) → voxelize(0.05 m) → spconv MinkUNet U-Net → 점별 feature
  ├ Semantic head → class logits   [CE + Lovász]
  ├ Center head   → centerness     [MSE]
  └ Offset head   → 3D offset       [L1, thing 점만]
→ thing 점을 offset만큼 이동 → 클래스별 DBSCAN → instance → panoptic merge
```
타깃·손실·그룹핑·merge의 정확한 수식은 [`DESIGN.md`](DESIGN.md) §2.1, 평가 수식은 §4.

## Results (val seq 08)
**실험 설정** (측정 시 사실 그대로 기입): train 시퀀스=`…`, voxel=`…` m, epoch=`…`, batch=`…`, precision=32.
공개 baseline은 **재학습이 아니라 논문 공개 수치를 인용**한 값이라 직접 비교 대상이 아니다. 설정 차이만
사실로 적고, 격차의 원인 해석은 결과가 나온 뒤 Discussion에서 근거를 갖고 한다.

**진행형 기여 분해(progressive) — 각 헤드가 지표에 미치는 영향**

| 모델 | mIoU | PQ | PQ† | FPS |
|---|---|---|---|---|
| Semantic only | _측정중_ | — | — | _측정중_ |
| + Center head | _측정중_ | _측정중_ | _측정중_ | _측정중_ |
| + Offset head (full) | _측정중_ | _측정중_ | _측정중_ | _측정중_ |

**공개 baseline 비교 (val seq 08, 논문 인용치)**

| 방법 | PQ | PQ† | SQ | RQ | mIoU |
|---|---|---|---|---|---|
| Panoptic-PolarNet (CVPR'21) | 59.1 | 64.1 | 78.3 | 70.2 | 64.5 |
| DS-Net (CVPR'21) | 57.7 | 63.4 | 77.6 | 68.0 | 63.5 |
| KPConv + PV-RCNN | 51.7 | 57.4 | 78.9 | 63.1 | 63.1 |
| PointGroup | 46.1 | 54.0 | 74.6 | 56.6 | 55.7 |
| **Ours** (spconv MinkUNet + center/offset, 축소) | _측정중_ | _측정중_ | _측정중_ | _측정중_ | _측정중_ |

DS-Net·Panoptic-PolarNet이 우리와 같은 bottom-up 계열이라 가장 가까운 비교 대상이다. (참고: 최신
Panoptic-PHNet test PQ 61.5.) 수치 출처는 아래 참고문헌.

### Oracle 디버깅 — PQ 오차를 semantic vs grouping으로 분해
구성 요소를 GT로 대체해 오차의 출처를 분리한다.

| 설정 | PQ | 무엇을 재나 |
|---|---|---|
| full (예측 semantic + 예측 grouping) | _측정중_ | 실제 성능 |
| `oracle=semantic` (GT semantic + 우리 offset/DBSCAN) | _측정중_ | grouping이 낼 수 있는 PQ 상한 |
| `oracle=instance` (예측 semantic + GT instance) | _측정중_ | semantic이 허용하는 PQ 상한 |

읽는 법: **full ↔ oracle=semantic** 격차 = *grouping/offset* 기여, **full ↔ oracle=instance** 격차 =
*semantic* 기여. 어느 쪽이 병목인지 수치로 말할 수 있다.
```bash
python -m src.eval ckpt=<best.ckpt> task=panoptic oracle=semantic data.root=$DATA
python -m src.eval ckpt=<best.ckpt> task=panoptic oracle=instance data.root=$DATA
```

## Qualitative
`scripts/viz`가 semantic / instance / panoptic 렌더를 만든다. 데모 프레임과 GIF는 학습 후 여기에 추가한다.
<!-- ![qualitative](demo/panoptic.gif) -->

**학습 곡선** (train loss ↓ / val mIoU ↑) — CSVLogger의 `metrics.csv`에서 생성:
`python -m scripts.plot_metrics <metrics.csv> demo/training_curves.png`
<!-- ![curves](demo/training_curves.png) -->

## Failure analysis
정성·정량 실패 모드를 [`docs/failure-analysis.md`](docs/failure-analysis.md)에 정리한다(학습 후 그림 포함).
관찰 예정 항목(가설):
- 인접한 사람들이 하나로 **merge** — offset이 두 중심 사이로 평균화될 때.
- **bicycle/bicyclist**의 offset이 불안정 — 얇고 점이 적어 중심 추정이 어려움.
- 작은 **traffic-sign**은 center heatmap이 약함 — 점 수가 적어 heatmap 신호가 낮음.

## Discussion
학습 결과가 나오면 아래 축으로 해석한다(측정 후 확정):
- **semantic mIoU는 거의 유지되는데 PQ가 오르는가?** instance 헤드가 semantic feature를 크게 흔들지 않는지,
  PQ 이득이 SQ(마스크 품질)에서 오는지 RQ(검출)에서 오는지 per-class로 분해.
- **어느 클래스가 PQ를 끌어내리나** — 작은 thing(bicycle/motorcycle/traffic-sign) 위주인지.
- **clustering 민감도** — DBSCAN `eps`에 따른 over/under-segmentation(ablation A2/A5와 연결).
- **latency / memory** — network vs end-to-end(DBSCAN 포함) FPS, voxel/배치에 따른 VRAM.

## Ablations
가설을 사전 등록하고 결과를 채운다 — [`ablations.md`](ablations.md). config 플래그로 실행:
`loss.offset=0` / `loss.center=0`(손실 기여), `cluster.eps=…`(추론만, 재학습 불필요), `data.voxel=0.10`,
`model.cr=0.5`.

## 직접 구현한 것 vs 가져다 쓴 것
직접 작성: SemanticKITTI 데이터셋·label remap·범위 crop·voxelize/collate(`src/data/`), spconv MinkUNet
U-Net(`src/models/backbone.py`), semantic·instance 타깃/손실(`src/lit_module.py`), offset-shift DBSCAN
(`src/panoptic/cluster.py`), 공식 evaluator를 감싼 PQ 어댑터(`src/panoptic/pq.py`), per-class 테이블+FPS
평가, Open3D viz, Hydra/Lightning/Docker.
가져다 쓴 것: spconv(sparse conv 커널), PRBonn `PanopticEval`(PQ 수식, vendor), 아키텍처 패턴
(Panoptic-DeepLab / DS-Net, MinkUNet), SemanticKITTI `learning_map`.

## 엔지니어링 노트
실제 LiDAR·batch>1로 규모를 올릴 때의 함정 정리 — [`docs/engineering-notes.md`](docs/engineering-notes.md)
(int32 좌표 flatten 한계, 좌표 열 순서 규약, spconv sm_86 SIGFPE, clean-clone 재현성 등).

## 재현
학습은 대여 클라우드 GPU(24GB+, ~170GB 디스크). spconv는 프리빌트 휠(소스 빌드 없음).
```bash
# 환경: docker build -f docker/Dockerfile -t panoptic .   또는   bash scripts/setup_cloud.sh
bash scripts/download_semantickitti.sh /data/semantickitti     # ~80GB
DATA=/data/semantickitti/dataset
python -m src.train task=semantic model=minkunet data.root=$DATA        # GATE 1
wget -O src/panoptic/np_ioueval.py \
  https://raw.githubusercontent.com/PRBonn/semantic-kitti-api/master/auxiliary/np_ioueval.py
python -m src.train task=panoptic  model=minkunet data.root=$DATA       # GATE 2
python -m src.eval  ckpt=<best.ckpt> task=panoptic data.root=$DATA      # PQ/mIoU + per-class + FPS
python -m scripts.viz ckpt=<best.ckpt> viz.frame=000100 viz.save=demo/ data.root=$DATA
# GPU/데이터 없이: PYTHONPATH=. python -m scripts.smoke_test
# 축소 설정: data.voxel=0.10 trainer.max_epochs=15 trainer.limit_train_batches=0.5
```

## 구조 · 계획
```
configs/  src/{data,models,panoptic,viz}  scripts/  docs/  DESIGN.md  ablations.md
```
빌드 순서와 go/no-go gate는 [`TASKS.md`](TASKS.md), 시간축/멀티뷰 확장 설계는
[`docs/consistency-4d.md`](docs/consistency-4d.md).

## 참고문헌 (비교 수치 출처)
- Panoptic-PolarNet, CVPR 2021 — [arXiv:2103.14962](https://arxiv.org/abs/2103.14962)
- DS-Net (Dynamic Shifting Network), CVPR 2021 — [arXiv:2011.11964](https://arxiv.org/abs/2011.11964)
- Panoptic-PHNet, CVPR 2022 — [arXiv:2205.07002](https://arxiv.org/abs/2205.07002)
