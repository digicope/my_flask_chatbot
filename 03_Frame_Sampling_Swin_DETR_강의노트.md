# 03 Frame Sampling & Swin/DETR — 강의용 노트

> **대상:** Video AI를 처음 접하는 초보자 (02 IPAD Temporal Pattern 수강 후)  
> **원본:** `03_Frame_Sampling_Swin_DETR.ipynb`  
> **실습 데이터:** `IPAD_sample.zip` → `IPAD_Sample/` (R01, 클립 01)

---

## 강의 전체 흐름

| 섹션 | 주제 | 한 줄 요약 |
|------|------|-----------|
| 0 | 환경 및 IPAD | 프레임 시퀀스 로드, 출력 경로 설정 |
| 1 | IPAD 데이터 탐색 | 설비 클립 메타·프레임 미리보기 |
| 1.5 | 디코딩 병목 | OpenCV vs Decord, 배치 로딩 벤치마크 |
| 2 | 프레임 샘플링 | Uniform / Motion / Cycle-aware 등 6종 전략 |
| 2.5 | GPU 배치 로딩 | `IpadBatchReader`, Decord `get_batch` 패턴 |
| 3 | Swin Transformer | Cycle-aware 프레임 Spatial 분류 |
| 4 | DETR | Motion top-k 프레임 객체 검출 |
| 5 | 통합 파이프라인 | 샘플링 → Swin + DETR → JSON 리포트 |

---

## 0. 환경 및 데이터 준비

### 핵심 용어

- **Frame Sampling(프레임 샘플링):** 긴 영상에서 **일부 프레임만** 골라 모델에 넣는 기법. 연산량·메모리 절약
- **Throughput(스루풋):** 초당 처리량. 디코딩 병목을 줄이면 최대 약 7배 향상(교재)
- **Decord(디코드):** MXNet 팀의 **딥러닝용 고속 비디오 디코더**. GPU(NVDEC) 지원, `get_batch`로 임의 프레임 일괄 로드
- **Swin Transformer(스윈 트랜스포머):** **S**hifted **Win**dow Attention. 계층적 Window Attention 2D ViT
- **DETR(디터, DEtection TRansformer):** Transformer Encoder-Decoder **end-to-end 객체 검출**. NMS 없이 query로 박스 예측
- **ViT(브이아이티, Vision Transformer):** 이미지를 패치 단위 token으로 Transformer 처리

### 소스코드 핵심

| 변수/함수 | 역할 |
|-----------|------|
| `SWIN_NUM_FRAMES=8`, `DETR_NUM_FRAMES=6` | 모델별 입력 프레임 수 |
| `SWIN_MODEL_ID` | `microsoft/swin-tiny-patch4-window7-224` |
| `DETR_MODEL_ID` | `facebook/detr-resnet-50` (COCO 91 classes) |
| `OUTPUT_DIR` | `output_sampling/` — figures, sampled_frames, manifest |

---

## 1. IPAD 데이터셋 탐색

### 핵심 용어

- **IpadClip(아이패드 클립):** 설비·split·clip_id·프레임 경로를 묶은 dataclass
- **Temporal 축(템포럴):** `000.jpg`, `001.jpg` … 파일명 정렬로 **시간 순서** 복원

### 소스코드

- `resolve_ipad_session()` → train 클립 1개
- `read_frames_bgr(clip, indices)` — 선택 인덱스만 읽기
- 저장: `01_ipad_preview.png` (시작·중간·끝 3프레임)

> **강의 포인트:** 이후 모든 분석은 **프레임 인덱스** 기반. 데이터 로딩이 정확해야 시간축·재현성이 보장됩니다.

---

## 1.5 디코딩 병목 현상

### 병목 3요소

| 병목 | 원인 | 대응 |
|------|------|------|
| **CPU-Bound Decoding(씨피유 바운드 디코딩)** | 프레임 디코딩이 CPU에서 순차 수행 | 배치 인덱스 로딩 |
| **I/O Overhead(아이오 오버헤드)** | 프레임 데이터 대용량 | 필요한 인덱스만 읽기 |
| **Inference Gap(인퍼런스 갭)** | 모델은 빠른데 데이터 공급 느림 | 샘플링 + 배치 전처리 |

### 라이브러리 비교

| Library | GPU Decoding | Random Access | Use Case |
|---------|-------------|---------------|----------|
| **OpenCV(오픈씨브이)** | 제한적 | 느림 | 범용 전처리 |
| **Decord(디코드)** | 높음(NVDEC) | 매우 빠름 | 대규모 학습 |
| **PyTorchVideo(파이토치비디오)** | 통합형 | 최적화됨 | PyTorch 파이프라인 |

### 소스코드

| 함수 | 역할 |
|------|------|
| `benchmark_opencv_sequential()` | 0번부터 전부 순차 읽기 |
| `benchmark_opencv_random()` | 임의 인덱스 하나씩 읽기 (seek 비용) |
| `benchmark_opencv_batch()` | 선택 인덱스만 배치 읽기 |
| `benchmark_decord_batch()` | mp4 + Decord GPU/CPU `get_batch` |
| `clip_to_mp4()` | IPAD jpg → mp4 변환 (Decord 시연용) |

---

## 2. 프레임 샘플링 자동화

### 핵심 용어

- **Uniform(유니폼, 균등):** 전체 구간 균등 추출 → Global 맥락 요약
- **Consecutive(컨시큐티브, 연속):** 특정 시점부터 연속 N프 → Action·모션 추적
- **Strided(스트라이디드, 간격):** stride 간격 추출 → 연산량·시간 범위 균형
- **Motion top-k(모션 탑케이):** Motion Energy 상위 구간 + 양끝 프레임
- **Cycle-aware(사이클 어웨어):** Autocorr 주기 T 기반 사이클별 샘플링 (산업 IPAD)
- **FPS downsampling(에프피에스 다운샘플링):** target FPS로 다운샘플 → 스트림 시뮬레이션

### `FrameSampler` / `SamplingStrategy`

```python
UNIFORM, CONSECUTIVE, STRIDED, MOTION_TOPK, CYCLE_AWARE, FPS
```

- `estimate_period_frames(motion)` — 02번과 동일 Autocorr 아이디어
- `export_sampled_clip()` — 샘플링 결과 jpg + manifest
- `sampling_manifest.json` — swin/detr preset 인덱스 통합 저장

### 프리셋

| 모델 | 샘플링 | 프레임 수 |
|------|--------|-----------|
| Swin | Cycle-aware | 8 |
| DETR | Motion top-k | 6 |

---

## 2.5 GPU 배치 로딩

### 4단계 워크플로 (교재)

1. 비디오 GPU 컨텍스트 로드  
2. 샘플링 인덱스 배치 구성  
3. `get_batch(indices)` 한 번에 디코딩  
4. PyTorch tensor 변환  

### `IpadBatchReader`

- `get_batch(indices)` → `(N,H,W,C)` RGB uint8
- `get_batch_tensor()` → `(N,C,H,W)` float32, device 이동

---

## 3. Swin Transformer — Spatial 분류

### 핵심 용어

- **Window Attention(윈도우 어텐션):** 작은 윈도우 단위 attention → ViT 대비 연산량 감소
- **Shifted Window(시프티드 윈도우):** 윈도우를 shift해 패치 간 정보 교환
- **Logits(로짓):** 모델 raw 출력. softmax 전 점수
- **Cosine Similarity(코사인 유사도):** 프레임 간 feature 벡터 방향 유사도

### 추론 흐름

```
Cycle-aware 8프 RGB → AutoImageProcessor → Swin-T → Top-k + logits cosine heatmap
```

> **강의 포인트:** ImageNet Top-1 라벨 자체보다 **프레임 간 유사도가 갑자기 떨어지는지**가 산업 영상에서 더 중요한 신호입니다.

### 소스코드

- `load_swin_model()`, `swin_classify_frames()`, `plot_swin_results()`
- 저장: `03_swin_classification.png`

---

## 4. DETR — 객체 검출

### 핵심 용어

- **Object Detection(객체 디텍션):** bbox + class + score
- **Query(쿼리):** DETR이 학습하는 **객체 슬롯**. 각 query가 하나의 detection 후보
- **NMS(엔엠에스, Non-Maximum Suppression):** 겹치는 박스 제거 — DETR은 **불필요**(end-to-end)
- **COCO(코코):** 80+ 클래스 일반 객체 데이터셋. 산업 설비와 불일치 → **오탐 가능**

### 추론 흐름

```
Motion top-k 6프 → DetrImageProcessor → DetrForObjectDetection
→ post_process_object_detection → bbox 시각화
```

> **실무:** COCO pretrained는 `toothbrush` 등 **엉뚱한 라벨** 가능 → **도메인 fine-tune** 필요

### 소스코드

- `detect_objects()`, `draw_detr_boxes()`, `plot_detr_gallery()`
- `DETR_SCORE_THRESHOLD = 0.35`

---

## 5. 통합 파이프라인

### `run_spatial_pipeline()`

1. Swin/DETR 프리셋 샘플링  
2. 프레임 export + manifest  
3. Swin 추론 (옵션)  
4. DETR 추론 (옵션)  
5. `pipeline_{device}_{clip}.json` 저장  

---

## Swin vs DETR 비교

| | Swin-T (2D) | DETR |
|---|-------------|------|
| 과제 | Image Classification | Object Detection |
| 샘플링 | Cycle-aware 8f | Motion top-k 6f |
| 산업 적용 | feature drift 관찰 | 부품 검출 (fine-tune 필요) |

---

## 시리즈 연계

| 노트북 | 역할 |
|--------|------|
| 01 | Video Transformer / 3D CNN |
| 02 | IPAD Temporal Pattern |
| 03 | Video 처리 최적화 + Spatial(Swin/DETR) |

---

## 부록: 출력 파일

| 경로 | 내용 |
|------|------|
| `output_sampling/figures/` | 벤치마크, 샘플링, Swin, DETR 그래프 |
| `output_sampling/sampled_frames/` | export된 샘플 프레임 |
| `output_sampling/sampling_manifest.json` | 통합 샘플링 manifest |
| `output_sampling/pipeline_*.json` | 파이프라인 리포트 |

---

*본 문서는 `03_Frame_Sampling_Swin_DETR.ipynb` 강의용으로 작성되었습니다.*
