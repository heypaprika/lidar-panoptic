# 희소 포인트클라우드 기반 Panoptic Segmentation

SemanticKITTI에서 LiDAR **panoptic segmentation**을 수행합니다. sparse-voxel **semantic** 백본
(spconv 기반 MinkUNet-style U-Net)에 가벼운 **center + offset** instance 헤드(bottom-up,
Panoptic-DeepLab / DS-Net 계열)를 얹어 확장했습니다. semantic 백본 → 점별 instance 중심 offset과
center heatmap → offset-shift 후 clustering → **panoptic 출력**이며, **공식** Panoptic Quality(PQ)와
mIoU로 평가합니다.

<!-- DEMO: scripts.viz가 demo/를 만들면 여기에 결과 이미지:
![semantic vs panoptic](demo/08_000100_panoptic.png)
-->

> 범위: SemanticKITTI에서 sparse-voxel semantic 백본을 panoptic으로 end-to-end 확장하고, 공식 지표
> (PQ/mIoU)·ablation·재현 가능한 학습 인프라를 갖춘다.

## 결과 (val seq 08)
축소 설정(아래 명시) — 짧은 컴퓨트 예산에서의 정직한 트레이드오프입니다. reference 행은 비교용 공개 수치.

| 설정 | mIoU | PQ | PQ† | SQ | RQ | FPS |
|---|---|---|---|---|---|---|
| semantic (spconv MinkUNet) | _측정중_ | — | — | — | — | _측정중_ |
| + center/offset panoptic | — | _측정중_ | _측정중_ | _측정중_ | _측정중_ | _측정중_ |
| _reference (공개 수치, 근사)_ | _~63_ | _~55–58_ | — | — | — | — |

> 설정: <시퀀스 / voxel / epoch>. 공개 수치와의 격차는 method가 아니라 주로 **epoch 수·voxel 해상도**에서
> 온다고 보며, 측정 후 정확한 차이를 기입합니다. reference: SPVCNN mIoU ≈ 63(mit-han-lab/spvnas),
> bottom-up panoptic PQ ≈ 55–58(DS-Net, Panoptic-PolarNet). ablation은 [`ablations.md`](ablations.md).

## 설계 이유
- **다시 만들지 말고 재사용**: semantic baseline을 먼저 재현(go/no-go gate)한 뒤 같은 백본에 instance 헤드를
  붙인다. gate는 `TASKS.md`, 수식은 `DESIGN.md`.
- **center + offset (embedding + MeanShift가 아니라)**: bottom-up 회귀는 조밀하고 잘 정의돼 있어 학습이
  안정적이고 non-trivial PQ에 빨리 도달한다. metric-learning embedding은 margin/bandwidth에 민감해
  **ablation(A1)**으로 남겨 두 접근을 비교한다.
- **PQ는 직접 구현하지 않는다**: >0.5 IoU 매칭과 void 처리가 미묘해 틀리기 쉬워, SemanticKITTI 공식
  evaluator를 감싸기만 한다.
- **지연시간 보고**: panoptic은 돌아가야 의미가 있다 — eval이 network·end-to-end FPS를 함께 출력한다.

## 직접 구현한 것 vs 가져다 쓴 것
이 레포에서 직접 작성:
- SemanticKITTI 데이터셋 + label remap, 범위 crop, 순수 numpy voxelize/collate (버전 무관, batch-first
  좌표, 비음수 시프트) — `src/data/`
- 점별 출력(devoxelize)까지 배선한 spconv MinkUNet U-Net, Native algo — `src/models/backbone.py`
- semantic(CE + Lovász)과 **instance 타깃/손실**(per-(scan,inst) centroid → offset_gt, Gaussian
  center_gt; MSE + 마스킹 L1) — `src/lit_module.py`, `src/losses.py`
- offset-shift **DBSCAN** clustering → panoptic merge — `src/panoptic/cluster.py`
- 공식 evaluator를 감싼 PQ **어댑터**(PQ† 포함) — `src/panoptic/pq.py`
- per-class 테이블 + FPS 평가, Open3D 시각화, Hydra/Lightning/Docker 인프라, gate 기반 계획

가져다 쓴 것(의존성, 직접 구현이라 주장하지 않음):
- **spconv**(sparse conv 커널), **PRBonn/semantic-kitti-api** `PanopticEval`(PQ 수식, vendor)
- 아키텍처 패턴: Panoptic-DeepLab / DS-Net(center+offset), MinkUNet(백본 형태)
- SemanticKITTI `learning_map`(raw→train class id)

## 엔지니어링 노트
toy에서 실제 LiDAR·batch>1로 규모를 올릴 때 물리는 비자명한 문제들을 정리했습니다 —
[`docs/engineering-notes.md`](docs/engineering-notes.md). 예를 들면:
- 좌표 flatten의 **int32 한계**로 batch>2에서 illegal memory access → 점군 범위 crop으로 볼륨 bound
- sparse-conv **좌표 열 순서** 규약 오류가 에러 없이 다운샘플을 오염(nnz 폭주)시키는 문제
- spconv implicit-GEMM의 sm_86 SIGFPE → `ConvAlgo.Native` 회피
- clean clone 재현성 검증(`git archive`로 누락 파일 조기 발견)

## 상태
파이프라인이 end-to-end로 검증됐고 실데이터에서 **학습이 수렴**합니다(val 08 semantic mIoU 상승 중).
합성 스모크 테스트(`scripts/smoke_test.py`)와 실제 스캔 백본 점검(`scripts/debug_backbone.py`) 모두 통과.
남은 일: 축소 설정 런을 마무리하고 위 표의 수치 기입(GATE 1 mIoU → GATE 2 PQ).

## 설치
학습은 **대여 클라우드 GPU**(24GB+ VRAM, ~170GB 디스크)를 대상으로 합니다. spconv는 프리빌트 휠이라
소스 빌드가 없습니다.

```bash
# A) Docker:  docker build -f docker/Dockerfile -t panoptic .
# B) sudo 되는 CUDA 박스:
bash scripts/setup_cloud.sh                          # spconv 휠 포함 pip 설치 + 스모크 테스트
bash scripts/download_semantickitti.sh /data/semantickitti   # ~80GB velodyne+labels
#   -> configs/data/semantickitti.yaml 의 root: /data/semantickitti/dataset 로 지정
```
GPU/데이터가 없다면: `PYTHONPATH=. python -m scripts.smoke_test` 로 파이프라인을 합성 검증할 수 있습니다.

## 실행
```bash
DATA=/data/semantickitti/dataset
python -m src.train task=semantic model=minkunet data.root=$DATA          # GATE 1: mIoU
wget -O src/panoptic/np_ioueval.py \
  https://raw.githubusercontent.com/PRBonn/semantic-kitti-api/master/auxiliary/np_ioueval.py
python -m src.train task=panoptic model=minkunet data.root=$DATA          # GATE 2: +center/offset
python -m src.eval  ckpt=<best.ckpt> task=panoptic data.root=$DATA        # PQ/mIoU + per-class + FPS
python -m scripts.viz ckpt=<best.ckpt> viz.frame=000100 viz.save=demo/ data.root=$DATA

# 짧은 예산용 축소 설정(GATE 수치를 빨리 확보; 결과표에 설정을 명시):
python -m src.train task=semantic model=minkunet data.root=$DATA \
    data.voxel=0.10 trainer.max_epochs=15 trainer.limit_train_batches=0.5
```

## 구조
```
configs/      Hydra (data / model / trainer)
src/data/     SemanticKITTI 데이터셋 + label map + voxelize/collate
src/models/   spconv 백본 + semantic/center/offset 헤드
src/panoptic/ offset-shift clustering + 공식 PQ 어댑터
src/viz/      Open3D 렌더링            scripts/  train/eval/viz + smoke/debug 헬퍼
docs/         engineering-notes.md    DESIGN.md  설계·수식·트레이드오프
TASKS.md      gate 기반 계획          ablations.md  사전 등록 실험
```
