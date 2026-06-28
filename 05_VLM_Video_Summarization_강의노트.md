# 05 VLM Video Summarization — 강의용 노트

> **대상:** Video AI 초보자  
> **원본:** `05_VLM_Video_Summarization.ipynb`  
> **실습 데이터:** `IPAD_sample.zip` (R01, 클립 01)

---

## 강의 전체 흐름

| Part | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1 | 정적 vs 동적 요약 | Keyframe · Video Skimming |
| 2 | VLM 융합 | CLIP · Spatiotemporal Projector |
| 3 | 적응형 키프레임 | AKS · T* · K-Medoids |
| 4 | 통합 파이프라인 | manifest · VLM 입력 카드 |

---

## Part 1 — 정적 vs 동적 요약

### 핵심 용어

- **Keyframe(키프레임):** 영상을 대표하는 **핵심 한 장**. 정적 요약
- **Video Skimming(비디오 스키밍):** 중요 **구간 클립**을 이어 붙인 동적 요약
- **Color Histogram(컬러 히스토그램):** 프레임 색 분포 — 비슷한 장면은 histogram 유사
- **BHATTACHARYYA(바타차리야):** histogram 거리
- **SSIM(에스에스아이엠, Structural Similarity):** 구조적 유사도. MSE보다 지각적으로 자연스러움
- **MSE(엠에스이, Mean Squared Error):** 픽셀 차이 제곱 평균 (SSIM fallback)
- **Shot Boundary(샷 바운더리):** 장면·카메라 컷 경계
- **Motion Energy(모션 에너지):** 프레임 차분 → 움직임 peak

### 정적 키프레임 3종

| 방법 | 원리 |
|------|------|
| Histogram | BHATTACHARYYA 거리 greedy 다양성 |
| SSIM/MSE | 구조적 차이 greedy |
| Motion Peak | `find_peaks` on Motion Energy |

### 동적 Skimming

```
Shot boundary → 구간 분할 → importance(motion) → 상위 35% 구간 선택
```

### 소스코드

- `select_keyframes_histogram/ssim/motion_topk()`
- `detect_shot_boundaries()`, `build_skim_timeline()`
- 저장: `01_static_keyframes.png`, `02_dynamic_skim_timeline.png`

> **한 줄:** 정적="어떤 **프레임**?", 동적="어떤 **구간**?"

---

## Part 2 — VLM 융합 & 토큰화

### 핵심 용어

- **VLM(브이엘엠, Vision-Language Model):** 이미지+텍스트 함께 이해
- **CLIP(클립):** image·text 공동 embedding, relevance score
- **ViT(브이아이티):** Vision Transformer patch token
- **Spatiotemporal(스페이시오템포럴):** 공간+시간
- **Q-Former(큐 포머):** learnable query → visual token cross-attention
- **MLP(엠엘피):** visual feature → LLM token space projection
- **Token Budget(토큰 버젯):** VLM 입력 최대 프레임/토큰 수 (`NUM_VLM_FRAMES=8`)
- **Cross-Attention(크로스 어텐션):** N프레임 → K토큰 압축

### 3단계 (교재)

```
Spatial (CLIP ViT) → Spatiotemporal Projector → LLM Token Alignment
```

### VLM 아키텍처 비교

| Model | Token Strategy |
|-------|----------------|
| **Video-LLaVA(비디오-라바)** | Uniform + projection |
| **Video-LLaMA(비디오-라마)** | Visual + Audio 병렬 |
| **LLaVA-Next-Video(라바-넥스트-비디오)** | AnyRes + spatiotemporal pooling |

### 소스코드

- `encode_frames_clip()`, `score_frames_with_clip()`
- `SpatiotemporalProjector` — `(B,T,D) → (B,K,D)`
- `compress_frame_tokens()` — LLaVA-Next-Video 토큰 압축 개념

---

## Part 3 — 시간 세그멘테이션 & 적응형 키프레임

### 핵심 용어

- **Change-Point(체인지포인트):** histogram·motion Z 급변 → 구간 경계
- **K-Medoids(케이-메도이드):** 클러스터 중심 = **실제 프레임**(medoid)
- **Medoid(메도이드):** 다른 점과 거리 합 최소인 **실제 샘플**
- **AKS(Adaptive Keyframe Sampling):** relevance + **temporal coverage** 동시 최대화 (CVPR 2025)
- **Coverage(커버리지):** 키프레임이 시간축 전체에 **고르게** 분포
- **T*(티-스타, Temporal Search):** coarse grid → promising cell **zoom-in** (CVPR 2025)
- **Long-form VLM(롱폼 브이엘엠):** 장영상 VLM — 전처리(선별·압축) 필수

### 방법 비교

| | AKS | T* | K-Medoids |
|---|-----|-----|-----------|
| 기준 | CLIP relevance + coverage | grid zoom-in | 구간별 medoid |
| 입력 | relevance 배열 | relevance 배열 | histogram distance |

### 소스코드

- `segment_by_change_points()`, `kmedoids_indices()`
- `adaptive_keyframe_sampling()`, `temporal_grid_search()`
- 저장: `03_segmentation_keyframes.png`

---

## Part 4 — 통합 VLM 요약 파이프라인

### 파이프라인 흐름

```
IPAD 프레임 → Motion/Shot/Change-point → CLIP relevance
       → AKS 키프레임 → Projector(K tokens)
       → Rule-based 요약 (+ BLIP 캡션 선택)
       → keyframes/*.jpg + summary_manifest.json
```

### 핵심 용어

- **Manifest(매니페스트):** 분석 결과 JSON (`summary_manifest.json`)
- **Rule-based Summary(룰 베이스드):** LLM 없이 규칙으로 narrative 생성
- **BLIP(블립):** Bootstrapping Language-Image Pre-training — 이미지 캡션
- **VLM Input Card(브이엘엠 인풋 카드):** VLM에 들어갈 키프레임 + 메타 시각화

### 소스코드

- `run_vlm_summary_pipeline()` → `VideoSummaryResult`
- `export_summary_artifacts()`, `plot_vlm_input_card()`
- `SUMMARY_QUERY = "industrial machine operation and motion"`

---

## 부록: 출력 파일

| 파일 | 내용 |
|------|------|
| `output_vlm_summary/summary_manifest.json` | 키프레임·세그먼트·요약 텍스트 |
| `output_vlm_summary/keyframes/` | VLM 입력 JPG |
| `figures/04_vlm_input_card.png` | VLM 입력 카드 |

---

## 확장 실습

- `RUN_BLIP=True` — BLIP 캡션 생성
- `SUMMARY_QUERY`를 이상 상황 질의로 변경 → AKS/T* 비교

---

*본 문서는 `05_VLM_Video_Summarization.ipynb` 강의용으로 작성되었습니다.*
