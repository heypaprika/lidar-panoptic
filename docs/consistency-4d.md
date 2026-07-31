# 시간축/멀티뷰 일관성 — 논문 정리와 확장 설계

이 레포는 **단일 스캔** panoptic이다. 다음 방향인 "프레임/뷰 간 instance 일관성"을 두 대표 논문의 수식으로
정리하고, 우리 center+offset 파이프라인에서 어떻게 확장할지 설계한다.

## 0. 출처
- **PanSt3R** — NAVER LABS Europe, ICCV 2025 (Žust, Cabon 외). arXiv:2506.21348, github: naver/panst3r.
  멀티뷰 이미지 + 3D 재구성 기반 panoptic. **(이 레포의 LiDAR 파이프라인과는 접근이 달라 개념적으로 참고.)**
- **4D Panoptic LiDAR Segmentation** — Aygün 외, CVPR 2021 (TUM). arXiv:2102.12472. **네이버 아님.**
  SemanticKITTI 4D panoptic 태스크와 **LSTQ** 지표를 처음 제안. **(LiDAR-네이티브 → 우리 데모의 앵커.)**

---

## 1. 4D Panoptic LiDAR (LSTQ) — 구현 앵커

**태스크.** 연속 스캔에서 같은 instance가 같은 id를 유지(시공간 tube). 매 스캔 독립 세그가 아니라 시간축
association까지 평가.

**지표 — LSTQ (LiDAR Segmentation and Tracking Quality).** 두 요소의 기하평균:
```
LSTQ = sqrt( S_cls · S_assoc )
```
기하평균이라 semantic·association 둘 중 하나만 나빠도 크게 깎인다 → 둘 다 잘해야 함.

- **S_cls (분류/semantic, instance-agnostic):**
  ```
  S_cls = (1/C) · Σ_c IoU(c),   IoU(c) = |TP_c| / (|TP_c| + |FN_c| + |FP_c|)
  ```
- **S_assoc (시공간 association):** GT track별로, 겹치는 예측 track들의 TPA-가중 IoU를 합산:
  ```
  S_assoc = (1/|T|) · Σ_{t ∈ GT tracks} (1/|gt(t)|) · Σ_{s: s∩t ≠ ∅} TPA(s,t) · IoU_id(s,t)

  TPA(s,t) = |pr(s) ∩ gt(t)|                      # true positive associations (겹치는 점 수)
  IoU_id(s,t)  =  TPA / (TPA + FPA + FNA)          # id 기준 IoU
  ```
  > 정규화 상수/집합 정의의 정확한 형태는 논문 및 semantic-kitti-api 구현으로 최종 확인. 핵심은
  > **"겹치는 track쌍의 TPA로 가중한 id-IoU"** 라는 구조와, S_cls와의 **기하평균** 결합이다.

**방법(원논문).** 시간창 `{max(0,t−τ), …, t}` 스캔을 ego-motion으로 정합해 **4D 볼륨** 구성 → 점별로
**embedding ε · covariance Σ · objectness O** 예측 → 가우시안 클러스터링으로 instance 할당:
```
p̂_ij = 1/((2π)^{D/2} |Σ_i|^{1/2}) · exp( −½ (e_i − e_j)^T Σ_i^{-1} (e_i − e_j) )   # p̂_ij > 0.5 면 같은 instance
```
겹치는 창 사이는 **점 교집합 기반 greedy overlap 매칭(IoU ≥ 0.5)**으로 id를 전파한다.

---

## 2. PanSt3R (NAVER) — 개념 프레임

**아키텍처.** 멀티뷰 3D 재구성 백본 **MUSt3R**(DUSt3R의 멀티뷰 확장) + **DINOv2** 2D feature를 결합해
조인트 토큰 생성. **Mask2Former식** 마스크 예측:
- **모든 뷰가 공유하는 learnable query** `{q_j}` — 각 query가 뷰 전체에서 **같은 instance**를 타깃.
  → 이것이 멀티뷰 일관성의 핵심(별도 consistency loss 없이 **공유 query + 3D-aware feature**에서 창발).
- 마스크: `M_{j,n} = sigmoid(F_n · q_j^M)`. 분류는 open-vocab(query ↔ SigLIP text embedding cosine).

**손실(Mask2Former 프로토콜):**
```
L = λ_c · L_cls(focal) + λ_d · L_dice + λ_b · L_bce      # λ_c=2, λ_d=5, λ_b=5
```

**뷰 간 instance 선택 — QUBO(이차 무제약 이진 최적화)로 전역 최적화:**
```
u* = argmax_{u ∈ {0,1}^m}  Σ_i u_i Q_i  −  Σ_{i<j} u_i u_j Q_{ij}
Q_i  = Σ_k M_{i,k}                 # 커버리지(마스크 면적)
Q_{ij} = Σ_k min(M_{i,k}, M_{j,k}) # 겹침 페널티
```
겹치지 않으면서 커버리지를 유지하는 instance 집합을 전역 선택 → 휴리스틱 병합보다 멀티뷰 품질↑.

---

## 3. 두 접근의 공통 원리 & 우리 레포와의 다리

**공통 원리:** *"프레임/뷰마다 독립 세그가 아니라, 공유 표현으로 instance를 전역 일관되게 만든다."*
- PanSt3R = **공유 query(뷰 간)** + **QUBO 전역 선택**.
- 4D-LiDAR = **공유 embedding/center** + **시공간 clustering** + **창 간 overlap 전파**.

**우리 레포와의 관계.** 현재 single-scan center+offset은 4D-LiDAR 계열의 **τ=0 특수 케이스**다. offset 헤드가
이미 instance 중심으로의 회귀를 학습하므로, 시공간으로 올리는 것은 새 아키텍처가 아니라 **같은 헤드 위의
association 문제**다(DESIGN §9).

---

## 4. 데모 설계 (10일 실현 가능, 수식 충실)

원논문의 완전한 학습형 4D(embedding+covariance+objectness)는 스트레치다. 대신 **우리 offset 헤드 + KITTI
pose + overlap association**으로 LSTQ의 핵심을 실증한다:

1. **정합**: KITTI `poses.txt`(+calib)로 연속 `2..T` 스캔을 공통 좌표계로 정합.
2. **스캔별 panoptic**: 각 스캔을 현재 파이프라인으로 semantic + offset-shift DBSCAN → thing instance.
3. **창 간 association**: 겹치는 점의 **TPA 기반 overlap(IoU ≥ 0.5) greedy 매칭**으로 id 전파(4D-LiDAR와
   동일 원리, 학습형 embedding 없이).
4. **평가/시각화**: 축소 설정(몇 시퀀스·짧은 창)에서 **LSTQ** 측정, 또는 정성 시각화(같은 차량이 프레임을
   넘어 같은 색 id를 유지).

**스코프.** 우선 (offset + pose + overlap) 경량 버전으로 시작하고, 학습형 시공간 embedding/clustering은
후속으로 둔다. 목표는 작동하는 최소 4D 데모 + 명확한 확장 경로다.
