# TASKS — 계획과 go/no-go gate

핵심 = Week 1–5. 스트레치(sim-to-real) = Week 6–7. 마무리 = Week 8.
실패한 gate는 다음 단계를 **막습니다** — 넘어가기 전에 고칩니다.

## 지금 우선순위 — 코어 완결 (범위 확장보다 먼저)
범위를 넓히기(4D·sim-to-real) 전에, 아래 스토리를 먼저 끝낸다:
1. [x] semantic baseline **mIoU 54.8** 확정 (GATE 1) + 학습 곡선
2. [x] panoptic **PQ 45.2** 확정 (GATE 2) + per-class + FPS + 곡선
3. [x] **oracle 디버깅**: semantic 95.1 / instance 46.7 → 병목은 semantic
4. [x] **ablation**: DBSCAN eps sweep(0.3/0.6/1.0 → 44.5/45.2/45.3, 강건)
5. [x] **qualitative** (BEV semantic/panoptic 그림, 헤드리스 matplotlib 렌더) + failure 분석 문서

4D 시간축 확장은 설계(`docs/consistency-4d.md`)로만 두고, 구현은 코어 완결 이후.

## Week 1–2 · Semantic 백본 재현  ⛔ GATE 1
- [x] SemanticKITTI 데이터셋 + label map (`src/data/`) — remap/instance/mIoU **단위 검증**.
- [x] Voxelize + collate (numpy, 버전 무관) + DataModule. batch-first 좌표, 비음수 시프트, 범위 crop.
- [x] Semantic 학습 배선: CE + Lovász, val **mIoU**, Hydra/Lightning 엔트리.
- [x] 실행 가능한 백본: **spconv MinkUNet U-Net** (torchsparse에서 전환 — 엔지니어링 노트 참고).
- [x] **파이프라인 스모크 테스트**(`scripts/smoke_test.py`, dummy 백본, 합성 점) — collate / heads /
      Lovász / CE / IoU를 torch 2.4 + CUDA에서 end-to-end 검증.
- [x] **실제 스캔 백본 검증**(`scripts/debug_backbone.py`) + 학습 **수렴 확인**(val 08 mIoU 상승 중).
- [ ] **GATE 1 최종 수치**: val(seq 08)에서 semantic **mIoU** 재현·기록. Open3D로 스캔 sanity-check.
- [ ] (upgrade) MinkUNet → **SPVCNN**(spvnas vendor)으로 성능 향상.
- **GATE 1:** semantic mIoU가 *공개 수치에 근접*(축소 설정 OK). 아니면 → 멈추고 수정.

## Week 3–4 · Panoptic 헤드  ⛔ GATE 2
- [x] Center 헤드(centerness heatmap) + Offset 헤드(instance 중심으로의 3D offset).
- [x] Instance 타깃 + 손실: per-(scan,inst) centroid → offset_gt/center_gt; MSE(center),
      L1(offset, thing 점). `_instance_targets`/`_instance_loss` **스모크 검증**(grad 흐름).
- [x] Offset-shift + DBSCAN clustering → instance id, val에 배선(`_accumulate_panoptic`).
- [x] PQ 어댑터(`panoptic/pq.py`), 공식 PanopticEval 위에; PQ/PQ†/SQ/RQ/mIoU + lazy vendor.
- [ ] `np_ioueval.py`(PRBonn) **vendor** + 실행: val seq 08에서 **non-trivial PQ**(클라우드 필요).
- **GATE 2:** val에서 non-trivial **PQ**(공식 evaluator). 아니면 → 헤드/clustering 디버그.

## Week 5 · 평가 + 시각화
- [x] Eval 엔트리(`src/eval.py`): ckpt → val mIoU + (panoptic) PQ/PQ†/SQ/RQ, val loop 재사용.
- [x] Open3D 렌더러(`scripts/viz.py` + `viz/render.py`): semantic vs panoptic, 인터랙티브 또는
      오프스크린 PNG(헤드리스). 데모 프레임 생성엔 ckpt 필요.
- [x] per-class PQ/SQ/RQ/IoU 테이블 + FPS(network 전용 & clustering 포함 end-to-end), `src/eval.py`.

## Week 6–7 · Sim-to-Real (스트레치 — 뺄 수 있음)
- [ ] CARLA 합성 LiDAR + 자동 label export → 포인트클라우드 데이터셋.
- [ ] 소규모 실험: 합성 pretrain → SemanticKITTI fine-tune(또는 self-training).
- [ ] 막히면: README에 **설계된 실험 + 예비 수치**로 제출.

## Week 8 · 제출
- [x] 결과표 + ablation 골격: `README.md` 표 + `ablations.md`(가설 사전 등록).
- [ ] 클라우드 런에서 결과/ablation 수치 기입.
- [ ] README 정리, 아키텍처 그림, **데모 영상/gif**, 짧은 기술 글.
- [ ] Dockerfile로 train/eval 재현.

## 실행할 Ablation (≥1개) — 가설은 `ablations.md`에 사전 등록
- A1 center-offset **vs** instance-embedding+MeanShift.
- A2 clustering: DBSCAN **vs** dynamic-shift(DS-Net).
- A3 center 헤드: aux 전용 **vs** center-NMS 그룹핑.
- A4 백본 폭(SPVCNN cr 1.0 vs 0.5): PQ vs FPS 트레이드오프.
- A5 voxel 크기 0.05 vs 0.10 m.
