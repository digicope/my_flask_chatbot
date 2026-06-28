# 04 Equipment Anomaly Scene Detection — 강의용 노트

> **대상:** Video AI 초보자 (05 VLM 요약 선행 권장)  
> **원본:** `04_Equipment_Anomaly_Scene_Detection.ipynb`  
> **실습 데이터:** `IPAD_sample.zip` (training=정상, testing=이상 포함)

---

## 강의 전체 흐름

| Part | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1 | IPAD train/test | 정상 baseline vs 테스트 클립 |
| 2 | Motion · Appearance | 프레임 단위 이상 신호 |
| 3 | CLIP 임베딩 | 정상 centroid 대비 편차 |
| 4 | VLM 쿼리 대조 | 정상 vs 이상 텍스트 relevance |
| 5 | 세그멘테이션 · 융합 | change-point + 4신호 fused score |
| 6 | 통합 파이프라인 | anomaly_manifest + 갤러리 export |

---

## Part 1 — IPAD 정상 baseline & 테스트 클립

### 핵심 용어

- **Baseline(베이스라인):** training(정상) 클립으로 만든 **기준 분포**
- **Unsupervised(언슈퍼바이즈드, 비지도):** testing 클립 없으면 training으로 데모
- **VAD(브이에이디, Video Anomaly Detection):** 비디오 이상 탐지
- **frame_labels(프레임 라벨):** `frame_labels_{device}.npy` — 0=정상, 1=이상 (선택)

### 소스코드

```python
IPAD_ROOT, TRAIN_CLIP, TEST_CLIP = resolve_anomaly_session()
TRAIN_FRAMES = load_clip_frames(TRAIN_CLIP)
TARGET_CLIP = TEST_CLIP or TRAIN_CLIP
```

- `Motion baseline (train mean)` — 이후 Z-score 비교 기준

---

## Part 2 — 프레임 단위 이상 신호

### 핵심 용어

- **Motion Z-score(모션 지-스코어):** 정상 motion 분포 대비 편차
- **Appearance Change(어피어런스 체인지):** 연속 프레임 **색·밝기** 급변 (Bhattacharyya 거리)
- **Bhattacharyya(바타차리야):** 두 histogram 거리. 0=같음, 1=다름
- **HSV Histogram(에이치에스브이 히스토그램):** 색 분포 벡터 — 장면 전환 탐지
- **Sigmoid(시그moid):** Z-score → 0~1 이상 점수 매핑

### 두 신호 비교

| 신호 | 측정 | 잡아내는 것 |
|------|------|-------------|
| Motion | 움직임 편차 | 설비 동작 이상 |
| Appearance | histogram 급변 | 카메라 전환·조명·장면 변화 |

### 소스코드

- `compute_motion_zscore()`, `compute_appearance_change()`
- `score_motion_anomaly()`, `score_against_reference()`
- 경고선 `0.55` (시각화용)
- 저장: `01_frame_anomaly_signals.png`

---

## Part 3 — CLIP 임베딩 baseline

### 핵심 용어

- **CLIP(클립, Contrastive Language-Image Pre-training):** 이미지·텍스트 **공동 embedding** 공간
- **Centroid(센트로이드):** 정상 training 프레임 embedding **평균 벡터** — "정상" 의미 중심
- **Cosine Distance(코사인 디스턴스):** centroid와의 각도 거리. 멀수록 시각적으로 다름
- **Embedding(임베딩):** 512차원 feature vector (`openai/clip-vit-base-patch32`)

### 소스코드

- `encode_frames_clip()` — L2 정규화 image embedding
- `build_normal_baseline()` — train centroid
- `score_embedding_anomaly()` — test 편차 → 0~1
- 저장: `01b_embedding_anomaly.png`

> Motion/Appearance는 **픽셀**, CLIP은 **의미·장면** 관점 — **상호 보완**

---

## Part 4 — VLM 쿼리 대조

### 핵심 용어

- **Query Contrast(쿼리 콘트라스트):** 이상 query 유사도 − 정상 query 유사도
- **Relevance(레커런스):** 텍스트와 프레임의 CLIP cosine 유사도

### 쿼리 예시

```
NORMAL_QUERY  = "normal industrial machine operation cycle"
ANOMALY_QUERY = "industrial machine malfunction abnormal vibration stuck error"
```

### 소스코드

- `score_anomaly_query_contrast()`
- 저장: `01c_query_contrast.png`

---

## Part 5 — 시간 세그멘테이션 & 이상 장면

### 핵심 용어

- **Change-Point(체인지포인트):** histogram·motion Z 급변 → 구간 경계
- **VideoSegment(비디오 세그먼트):** `[start, end)` 연속 프레임 구간
- **Fusion(퓨전, 융합):** 4신호 가중 합산
- **AKS(Adaptive Keyframe Sampling):** 대표 이상 프레임 선별 (05번과 동일 개념)
- **Fused Threshold(퓨즈드 쓰레숄old):** 상위 20% percentile → 이상 프레임

### 융합 가중치

```
motion 30% + appearance 20% + embedding 35% + query 15%
```

### 대시보드 읽는 법

| 요소 | 의미 |
|------|------|
| 빨간 융합 곡선 | 최종 이상 점수 |
| 주황 음영 | 이상 **장면(구간)** |
| 히트맵 | 시간×신호, 밝을수록 이상 |
| ★ 별표 | AKS 대표 이상 프레임 |

### 소스코드

- `segment_by_change_points()`, `fuse_anomaly_scores()`
- `detect_anomaly_scenes()`, `adaptive_keyframe_sampling()`
- `plot_anomaly_overview()` — 4-in-1 대시보드
- 저장: `02_anomaly_overview.png`

---

## Part 6 — 통합 파이프라인

### `run_anomaly_detection_pipeline()`

→ `AnomalyDetectionResult` → `anomaly_manifest.json`

### 산출물

| 폴더/파일 | 설명 |
|-----------|------|
| `output_anomaly_scene/figures/` | 8장 분석 그래프 |
| `anomaly_frames/` | AKS 대표 JPG |
| `anomaly_scenes/` | 장면별 mid-frame |
| `anomaly_manifest.json` | 점수·구간·프레임 (알람 연동용) |

---

## 05 VLM 요약 vs 04 이상 탐지

| | 05 비디오 요약 | 04 이상 탐지 |
|---|---------------|-------------|
| Baseline | 전체 relevance | **정상 training centroid** |
| CLIP | summary query | **정상 vs 이상 contrast** |
| 산출물 | summary_manifest | **anomaly_manifest** |

---

## FAQ

**Q. GT 라벨 없어도 되나요?**  
A. 네. GT는 검증용 음영만, 탐지는 정상 baseline 대비로 동작합니다.

**Q. 이상이 잘 안 보여요.**  
A. `ANOMALY_PERCENTILE`을 75로 낮추거나 CLIP 가중치를 높이세요.

---

*본 문서는 `04_Equipment_Anomaly_Scene_Detection.ipynb` 강의용으로 작성되었습니다.*
