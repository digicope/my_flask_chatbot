# 01 Temporal Vision — 강의용 노트

> **대상:** Video AI를 처음 접하는 초보자  
> **원본:** `01_Temporal_Vision.ipynb`  
> **샘플 영상:** PyTorchVideo 공개 `archery.mp4` (양궁 동작 클립)

---

## 강의 전체 흐름

| 섹션 | 주제 | 한 줄 요약 |
|------|------|-----------|
| 0 | 환경 및 데이터 | 비디오를 읽고, 분석할 준비를 한다 |
| 1 | Spatiotemporal(시공간) 정보 | 공간(한 장면)과 시간(움직임)을 구분한다 |
| 2 | Optical Flow(광학 흐름) | 픽셀이 어디로 움직였는지 계산한다 |
| 3 | SlowFast(슬로우패스트) | 느린 경로 + 빠른 경로로 효율적으로 본다 |
| 4 | 3D CNN | 시간 축까지 포함한 합성곱으로 행동을 인식한다 |
| 5 | Video Transformer(비디오 트랜스포머) | Attention(어텐션)으로 시공간 패턴을 학습한다 |
| 6 | 모델 비교 요약 | Classical → 3D CNN → Transformer 진화를 정리한다 |

---

## 0. 환경 및 데이터 준비

### 이 섹션에서 배우는 것

- 비디오 AI 실습에 필요한 라이브러리와 샘플 영상을 준비한다.
- 비디오를 프레임 단위로 읽고, 모델 입력 형식으로 변환하는 **공통 도구 함수**를 이해한다.

### 핵심 용어

- **Temporal Vision(템포럴 비전):** 시간에 따라 변하는 영상(비디오)을 이해하는 컴퓨터 비전 분야
- **Spatiotemporal(스페이시오템포럴, 시공간):** Spatial(공간) + Temporal(시간)을 합친 말. "어디에 무엇이 있고, 어떻게 움직이는가"를 함께 다룬다
- **Frame(프레임):** 비디오를 잘게 나눈 한 장의 이미지. 영화 필름의 한 컷과 같다
- **FPS(에프피에스, Frames Per Second):** 1초에 몇 장의 프레임이 있는지. 숫자가 클수록 부드럽다
- **BGR(비지알):** OpenCV(오픈씨브이)가 기본으로 쓰는 색 순서 (Blue, Green, Red)
- **RGB(알지비):** 딥러닝 모델이 주로 쓰는 색 순서 (Red, Green, Blue)
- **PyTorch(파이토치):** 딥러닝 프레임워크
- **torchvision(토치비전):** PyTorch용 이미지·비디오 모델 라이브러리
- **Transformers(트랜스포머스):** Hugging Face(허깅페이스)의 사전학습 모델 라이브러리
- **Kinetics-400(키네틱스-400):** 400가지 사람 행동(걷기, 양궁 등)으로 학습된 대표 비디오 데이터셋

### 필요 패키지

```
opencv-python, numpy, matplotlib, torch, torchvision, transformers, Pillow
```

### 소스코드 핵심 기능

#### ① 전역 설정 (Cell 4)

| 변수/함수 | 역할 |
|-----------|------|
| `VIDEO_URL`, `VIDEO_PATH` | 샘플 영상 URL과 로컬 저장 경로 |
| `OUTPUT_DIR` 등 | 결과 이미지 저장 폴더 (`output/optical_flow`, `slowfast`, `spatiotemporal`) |
| `TIMESFORMER_NUM_FRAMES = 8` | TimeSformer(타임스포머)는 8프레임 입력 |
| `VIVIT_NUM_FRAMES = 32` | ViViT(비비트)는 32프레임 입력 |
| `R3D_NUM_FRAMES = 16` | R3D/MC3는 16프레임, 112×112 크기 |
| `SLOWFAST_ALPHA = 4` | SlowFast에서 Fast 경로가 Slow보다 4배 많은 프레임 사용 |
| `setup_matplotlib_korean()` | 그래프 한글 폰트(맑은 고딕 등) 자동 설정 |

> **강의 포인트:** 모델마다 필요한 프레임 수와 해상도가 다르다. "비디오를 통째로 넣는다"가 아니라 **고정 길이 클립(clip)** 으로 잘라 넣는다.

#### ② `download_sample_video()`

- Colab / Windows / Linux 환경에 맞게 `archery.mp4`를 다운로드한다.
- 이미 파일이 있으면 **재다운로드하지 않는다.**

#### ③ `read_all_frames_bgr()`

- OpenCV `VideoCapture`로 비디오 전체를 읽어 **BGR numpy 배열 리스트**로 반환한다.
- shape: `(높이, 너비, 3)`, dtype: `uint8` (0~255 정수)

#### ④ `sample_frame_indices()`

- 전체 프레임 중 **균등 간격**으로 N개 인덱스를 뽑는다.
- 예: 100프레임 영상에서 8프레임 → 0, 14, 28, … 99번 프레임 선택

#### ⑤ `read_sampled_rgb_frames()`

- 균등 샘플링 후 BGR → RGB 변환.
- 프레임이 부족하면 마지막 프레임을 반복해 길이를 맞춘다.

#### ⑥ `VideoMetadata` / `analyze_video_metadata()`

- 비디오의 **공간 정보**(가로×세로)와 **시간 정보**(FPS, 총 프레임, 재생 시간)를 추출한다.
- `frame_count / fps = duration_sec` (재생 시간 계산)

---

## 1. 시공간(Spatiotemporal) 정보

### 이 섹션에서 배우는 것

- 비디오를 **공간(Spatial)** 과 **시간(Temporal)** 두 관점으로 나누어 본다.
- 가장 단순한 시간 특징인 **프레임 차분**과 **Motion Energy(모션 에너지)** 를 계산한다.

### 핵심 용어

- **Spatial Context(스페이셜 컨텍스트, 공간 맥락):** 한 장의 프레임 안에서 보이는 것 — 사람 모양, 활 색깔, 배경 등 **정적(고정) 정보**
- **Temporal Context(템포럴 컨텍스트, 시간 맥락):** 프레임과 프레임 사이의 **변화** — 팔 움직임, 화살 발사 순서 등 **동적(움직임) 정보**
- **Frame Difference(프레임 디퍼런스, 프레임 차분):** 연속 두 프레임의 픽셀 값 차이 `|F(t+1) - F(t)|`. 차이가 크면 그 구역에 움직임이 있다
- **Motion Energy(모션 에너지):** 프레임 차분 값의 평균 등으로 "이 순간 얼마나 많이 움직였는지"를 숫자로 표현한 것
- **Grayscale(그레이스케일):** 흑백 이미지. 색 정보 없이 밝기만 사용 → 계산이 빠르다
- **CNN(씨엔엔, Convolutional Neural Network):** 2D 이미지의 공간 특징을 잘 뽑는 신경망
- **ViT(브이아이티, Vision Transformer):** 이미지를 패치 단위로 나눠 Attention으로 처리하는 모델

### 개념 비유 (초보자용)

| 관점 | 질문 | 양궁 영상 예시 |
|------|------|---------------|
| Spatial(공간) | "지금 화면에 **무엇**이 있나?" | 활, 과녁, 양궁 선수 |
| Temporal(시간) | "**어떻게** 변하나?" | 시위 당기기 → 발사 → 화살 비행 |

> **핵심:** 사진(1장)은 Spatial만, 비디오(여러 장)는 Spatial + Temporal을 함께 봐야 한다.

### 소스코드 핵심 기능

#### ① `visualize_spatiotemporal_overview()`

```
[상단] 시간 축을 따라 8개 프레임 격자 → Spatial 맥락
[하단 좌] 단일 프레임 → 정적 특징
[하단 우] |F(t+1)-F(t)| 차분 맵 → 동적 특징 (움직인 곳이 밝게)
```

- `cv2.absdiff()`: 두 그레이스케일 프레임의 절대 차분
- `cmap="hot"`: 움직임이 큰 곳을 밝은색(빨강·노랑)으로 표시

#### ② `compute_motion_energy_timeline()`

전체 비디오에 대해 두 가지 지표를 프레임마다 계산:

| 지표 | 계산 방법 | 의미 |
|------|-----------|------|
| Frame Diff Mean | `absdiff` 픽셀 평균 | 단순한 움직임 크기 |
| Flow Magnitude Mean | Farneback(파르네백) 광학 흐름 크기 평균 | 더 정교한 움직임 크기 |

#### ③ `plot_motion_energy_timeline()`

- 위 두 지표를 **시간 축 그래프**로 그려, "어느 순간 동작이 활발한지" 한눈에 본다.

#### ④ 섹션 1 실행 코드 (Cell 8)

```python
video_path = download_sample_video()
meta = analyze_video_metadata(video_path)          # 메타데이터 출력
visualize_spatiotemporal_overview(...)             # 공간 vs 시간 시각화
plot_motion_energy_timeline(...)                   # Motion Energy 그래프
```

---

## 2. 광학 흐름 (Optical Flow)

### 이 섹션에서 배우는 것

- **Optical Flow(옵티컬 플로우, 광학 흐름):** 연속 프레임에서 각 픽셀(또는 특징점)이 **어디로, 얼마나** 이동했는지 벡터로 표현하는 기법
- **Dense(덴스)** 방식(Farneback)과 **Sparse(스파스)** 방식(Lucas-Kanade)의 차이를 이해한다.

### 핵심 용어

- **Optical Flow(옵티컬 플로우, 광학 흐름):** "빛의 흐름"이라는 뜻. 실제로는 **픽셀 이동 벡터(화살표)** 를 추정하는 알고리즘 family
- **Farneback(파르너백):** Gunnar Farneback(군나르 파르너백)이 제안한 **Dense Optical Flow** 알고리즘. **모든 픽셀**의 (dx, dy) 이동량을 추정한다
- **Lucas-Kanade(루카스-카나데):** Bruce D. Lucas(루카스)와 Takeo Kanade(카나데)가 제안한 **Sparse Optical Flow** 알고리즘. **코너 특징점**만 골라 빠르게 추적한다
- **Dense(덴스):** 모든 픽셀에 대해 흐름 계산 → 정보는 풍부, 연산량 큼
- **Sparse(스파스):** 일부 특징점만 추적 → 빠르고 직관적
- **HSV(에이치에스브이):** Hue(색상), Saturation(채도), Value(밝기) 색 공간. 광학 흐름 **방향=색**, **크기=밝기**로 시각화
- **Hue(휴, 색상):** HSV에서 **각도(방향)** 를 색으로 표현
- **Value(밸류, 밝기):** HSV에서 **크기(속력)** 를 밝기로 표현
- **Shi-Tomasi(시-토마시):** 코너(모서리) 특징점을 찾는 알고리즘. `goodFeaturesToTrack`의 기반
- **PyrLK(피라엘케이, Pyramid Lucas-Kanade):** 이미지 피라미드(여러 해상도)를 이용한 Lucas-Kanade 추적. `calcOpticalFlowPyrLK`
- **Magnitude(매그니튜드, 크기):** 이동 벡터의 길이 = 움직임 속도
- **FlowNet / PWC-Net / RAFT(플로우넷 / 피더블유씨넷 / 래프트):** 딥러닝 기반 광학 흐름의 발전 단계 (Classical → Deep Learning → SOTA)

### Farneback vs Lucas-Kanade 비교

| | Farneback | Lucas-Kanade |
|---|-----------|--------------|
| 유형 | Dense | Sparse |
| 대상 | 모든 픽셀 | 코너 특징점 (~200개) |
| 시각화 | HSV 컬러맵 (방향=색, 크기=밝기) | 녹색 화살표 + 빨간 점 |
| 장점 | 전체 움직임 패턴 파악 | 빠르고 직관적 |

### Farneback 주요 파라미터

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| `pyr_scale=0.5` | 0.5 | 이미지 피라미드 축소 비율 (다단계 해상도) |
| `levels=3` | 3 | 피라미드 단계 수 (거친→세밀) |
| `winsize=15` | 15 | 주변 15×15 픽셀 영역을 보고 이동 추정 |

### 소스코드 핵심 기능

#### ① `compute_dense_optical_flow()`

```python
cv2.calcOpticalFlowFarneback(g0, g1, ...)
# 반환: (H, W, 2) — [:,:,0]=x방향, [:,:,1]=y방향 이동량
```

#### ② `flow_to_hsv_image()`

1. `cv2.cartToPolar()`: (dx, dy) → (magnitude, angle) 극좌표 변환
2. angle → Hue(0~179), magnitude → Value(0~255)로 HSV 이미지 생성
3. BGR로 변환해 화면에 표시

#### ③ `compute_sparse_optical_flow_lk()`

```
1단계: goodFeaturesToTrack → Shi-Tomasi 코너 최대 200개 검출
2단계: calcOpticalFlowPyrLK → 다음 프레임에서 점 추적
3단계: 추적 성공 점에 arrowedLine(화살표) + circle(끝점) 그리기
```

#### ④ `visualize_optical_flow_comparison()`

- 2×2 그리드: 이전/다음 프레임 | Farneback HSV | Lucas-Kanade 화살표

#### ⑤ 섹션 2 실행 코드 (Cell 11)

```python
frames = read_all_frames_bgr(video_path)
pair_idx = 10  # 10→11번 프레임 (양궁 동작 구간)
visualize_optical_flow_comparison(frames[10], frames[11], ...)
```

---

## 3. SlowFast Network — Dual-pathway(듀얼 패스웨이)

### 이 섹션에서 배우는 것

- **SlowFast(슬로우패스트):** 하나의 비디오를 **두 갈래 경로**로 나눠 처리하는 3D CNN 아키텍처
- Slow(느린) 경로는 **공간(고해상도)**, Fast(빠른) 경로는 **시간(저해상도·고프레임)** 을 담당한다.

### 핵심 용어

- **SlowFast(슬로우패스트):** Facebook AI Research(FAIR)가 제안한 이중 경로 비디오 인식 네트워크
- **Dual-pathway(듀얼 패스웨이, 이중 경로):** 하나의 입력을 두 개의 서로 다른 경로(Slow + Fast)로 처리
- **Slow Pathway(슬로우 패스웨이):** 프레임 수 적음(8프), **원본 고해상도** → "무엇인지"(객체·형태) 파악
- **Fast Pathway(패스트 패스웨이):** 프레임 수 많음(α=4배), **112×112 저해상도** → "어떻게 움직이는지" 파악
- **Alpha(알파, α):** Fast 경로의 프레임 밀도 배율. α=4이면 Slow 8프 → Fast 32프
- **P-cell / M-cell(피셀 / 엠셀):** 인간 망막의 두 종류 신경세포. P-cell=고해상도·저속, M-cell=저해상도·고속 → SlowFast의 생물학적 영감
- **3D CNN(쓰리디 씨엔엔):** 시간 축까지 포함한 3차원 합성곱 신경망 (섹션 4에서 상세)

### Slow vs Fast 비교

| | Slow Pathway | Fast Pathway |
|---|-------------|--------------|
| 프레임 수 | 8 (적음) | 32 (α=4배) |
| 해상도 | 원본 (고해상도) | 112×112 (저해상도) |
| 역할 | Spatial — 형태·세부 | Temporal — 빠른 동작 |
| 비유 | "사진을 자세히 본다" | "움직임을 빠르게 본다" |

> **강의 포인트:** 본 실습은 SlowFast **입력 구성 원리**를 시각화한다. 실제 torchvision SlowFast 추론은 별도 dual-tensor 파이프라인이 필요하다.

### 소스코드 핵심 기능

#### ① `build_slowfast_pathways()`

```
Slow: 전체에서 8프레임 균등 샘플링, 원본 해상도 유지
Fast: 32프레임 균등 샘플링, cv2.resize(112, 112)로 축소
```

#### ② `visualize_slowfast_pathways()`

- Slow / Fast 프레임을 각각 가로 모자이크(`np.hstack`)로 붙여 2행 그래프로 표시

#### ③ `print_architecture_comparison_table()`

- C3D, I3D, SlowFast, TimeSformer, ViViT, Swin3D 아키텍처를 한 표로 비교 출력

---

## 4. 3D CNN Architectures(쓰리디 씨엔엔 아키텍처)

### 이 섹션에서 배우는 것

- **3D CNN**으로 비디오 **행동 인식(Action Recognition)** 을 수행한다.
- R3D(C3D 계열), MC3(I3D 계열), Swin3D(Video Swin) 세 모델의 추론 결과를 비교한다.

### 핵심 용어

- **3D CNN(쓰리디 씨엔엔):** Conv3D(씨온브쓰리디) — 커널이 `(시간, 높이, 너비)` 3차원. **시간 패턴을 직접 학습**
- **2D CNN(투디 씨엔엔):** Conv2D — `(높이, 너비)`만. 프레임별 공간 특징만 추출
- **C3D(씨쓰리디):** 3D Convolution을 처음 비디오에 적용한 대표 모델 (2014)
- **R3D-18(알쓰리디-18):** ResNet-18(레즈넷-18) 구조를 3D Conv로 확장. torchvision에서 C3D 계열 대표
- **I3D(아이쓰리디, Inflated 3D):** 2D CNN(ImageNet 사전학습) 가중치를 3D로 **Inflation(인플레이션, 팽창)** 해 전이 학습
- **MC3-18(엠씨쓰리-18):** Mixed Convolution 3D. I3D 계열. 채널별 Conv3D 구조
- **Inflation(인플레이션):** 2D 필터를 시간 축 방향으로 복제·확장해 3D 필터로 만드는 기법. ImageNet 지식을 비디오에 전이
- **Swin3D(스윈쓰리디, Video Swin Transformer):** 3D Window Attention(윈도우 어텐션)으로 로컬 시공간 패치 내 Self-Attention
- **Self-Attention(셀프 어텐션):** 입력 패치들끼리 서로 "어디를 중요하게 볼지" 가중치를 계산
- **Action Recognition(액션 레코그니션, 행동 인식):** 비디오에서 "걷기", "양궁" 등 행동 클래스를 분류
- **Softmax(소프트맥스):** logits(로짓, 모델 출력 점수)를 0~1 확률로 변환. 합이 1
- **Top-k(탑케이):** 확률 상위 k개 클래스를 출력

### 3D CNN vs 2D CNN

```
2D Conv:  [높이 × 너비]         → 한 프레임의 공간만
3D Conv:  [시간 × 높이 × 너비]   → 여러 프레임의 시공간 패턴
```

### 모델 비교표

| Model | 교재 대응 | 핵심 원리 | 입력 |
|-------|-----------|-----------|------|
| R3D-18 | C3D | 3D Conv로 시공간 동시 학습 | 16프 × 112² |
| MC3-18 | I3D | 2D→3D Inflation, ImageNet 전이 | 16프 × 112² |
| Swin3D-T | Video Swin | 3D Window Attention | 32프 × 224² |

### 전처리 파이프라인

```
RGB 프레임 (T,H,W,C)
  → numpy stack → PyTorch TCHW tensor
  → weights.transforms() (리사이즈·크롭·정규화)
  → CTHW float32
  → 모델 입력
```

- **TCHW / CTHW:** Tensor 차원 순서. T=Time, C=Channel, H=Height, W=Width

### 소스코드 핵심 기능

#### ① `_prepare_torchvision_clip()`

- OpenCV RGB 프레임 → `(T,C,H,W)` uint8 tensor → torchvision transform → `(C,T,H,W)` float32

#### ② `predict_torchvision_video_model()`

```
1. clip 준비 (배치 차원 추가 → shape: 1,C,T,H,W)
2. model.eval() + torch.no_grad() → 추론
3. F.softmax → top-k 클래스명 + 확률 반환
```

#### ③ `run_3d_cnn_benchmark()`

- R3D-18, MC3-18, Swin3D-T 순차 로드·추론
- `RUN_SWIN3D=False`이면 Swin3D 건너뜀 (가중치 ~122MB)

#### ④ `print_predictions()`

- Top-5 행동 라벨과 확률(%)을 보기 좋게 출력

---

## 5. Video Transformer(비디오 트랜스포머)

### 이 섹션에서 배우는 것

- CNN 대신 **Transformer(트랜스포머)** 의 Attention 메커니즘으로 비디오를 이해한다.
- **TimeSformer(타임스포머)** 와 **ViViT(비비트)** 의 Space-Time Attention 방식 차이를 이해한다.

### 핵심 용어

- **Transformer(트랜스포머):** "Attention is All You Need" 논문의 Self-Attention 기반 신경망. NLP(자연어)에서 시작, Vision(비전)으로 확장
- **TimeSformer(타임스포머):** Facebook이 제안. **Divided Space-Time Attention(디바이디드 스페이스-타임 어텐션)** — Spatial Attention → Temporal Attention 순차 분리
- **ViViT(비비트, Video Vision Transformer):** Google이 제안. **Tubelet(튜블릿)** 임베딩 + **Factorized Attention(팩터라이즈드 어텐션)**
- **Tubelet(튜블릿):** 연속 2프레임 × 16×16 패치를 하나의 3D 토큰으로 묶은 단위. "작은 시공간 덩어리"
- **Token(토큰):** Transformer에 입력되는 최소 단위 (텍스트의 단어, 비전의 패치/튜블릿)
- **Patch(패치):** 이미지를 잘게 나눈 조각 (예: 16×16)
- **Attention(어텐션):** "어떤 부분에 집중할지" 가중치를 학습하는 메커니즘
- **AutoImageProcessor(오토이미지프로세서):** Hugging Face의 이미지/비디오 전처리기 (리사이즈, 정규화 자동)
- **Logits(로짓):** 모델의 raw 출력 점수 (아직 확률 아님)

### Space-Time Attention 3가지

| 방식 | 설명 | 대표 모델 | 효율 |
|------|------|-----------|------|
| **Joint Attention(조인트 어텐션, ST)** | 모든 시공간 패치에 한 번에 Attention | — | 연산량 최대 |
| **Divided Attention(디바이디드, T+S)** | Spatial → Temporal 순차 분리 | TimeSformer | Joint 대비 ~10× 효율 |
| **Factorized Attention(팩터라이즈드)** | 시공간 축을 독립 Encoder로 분해 | ViViT | Tubelet 단위 특징 |

### TimeSformer vs ViViT

| | TimeSformer | ViViT |
|---|-------------|-------|
| Attention | Divided (T+S) | Factorized |
| 입력 | 8프 × 224² | 32프 × 224² |
| 핵심 | 공간 먼저, 시간 나중 | Tubelet(2×16×16) 3D 토큰 |
| 모델 ID | `facebook/timesformer-base-finetuned-k400` | `google/vivit-b-16x2-kinetics400` |

### TimeSformer 파이프라인

```
RGB 프레임 8장
  → AutoImageProcessor (224×224, mean/std 정규화)
  → pixel_values (1, 8, 3, 224, 224)
  → TimesformerForVideoClassification
  → logits (400,) → softmax → top-k
```

### 소스코드 핵심 기능

#### ① `predict_timesformer()`

1. 8프레임 RGB 샘플링
2. `AutoImageProcessor` + `TimesformerForVideoClassification` 로드
3. `model.eval()` + `torch.no_grad()` 추론
4. softmax → top-k Kinetics-400 라벨 반환

#### ② `predict_vivit()`

1. 32프레임 RGB 샘플링
2. `VivitImageProcessor` + `VivitForVideoClassification` 로드
3. Tubelet size `[2, 16, 16]` 정보 함께 출력
4. softmax → top-k 반환

#### ③ `print_space_time_attention_guide()`

- Joint / Divided / Factorized Attention 개념을 콘솔에 출력

---

## 6. 모델 비교 요약

### 이 섹션에서 배우는 것

- Classical → 3D CNN → Video Transformer **기술 진화 타임라인**을 정리한다.
- 동일 archery 샘플에 대해 **모든 모델의 Top-1 예측**을 한 표로 비교한다.

### Vision 기술 진화 타임라인

```
Optical Flow (픽셀 변위, 규칙 기반)
        ↓
3D CNN — C3D → I3D → SlowFast (시공간 Convolution)
        ↓
Video Transformer — TimeSformer / ViViT / Swin3D (Self-Attention)
```

### 단계별 Temporal 정보 추출 방식

| 단계 | 대표 기법 | Temporal 정보 추출 |
|------|-----------|-------------------|
| Classical(클래시컬) | Farneback, Lucas-Kanade | 픽셀/코너 이동 벡터 (규칙 기반) |
| 3D CNN | R3D, MC3, Swin3D | 3D Conv / 3D Window Attention |
| Transformer | TimeSformer, ViViT | Divided / Factorized Self-Attention |

### 아키텍처 한눈에 비교

| Architecture | Core Concept | Key Advantage |
|--------------|--------------|---------------|
| C3D / R3D-18 | 고정 3D Conv | 단순·빠른 시공간 학습 |
| I3D / MC3-18 | 2D 가중치 Inflation | ImageNet 지식 전이 |
| SlowFast | Slow+Fast 이중 경로 | 효율·정밀도 균형 |
| Swin3D | 3D Window Attention | 로컬 시공간 맥락 |
| TimeSformer | Divided Space-Time | Joint 대비 ~10× 효율 |
| ViViT | Tubelet + Factorized | 시공간 튜브 단위 특징 |

### 소스코드 핵심 기능

#### ① `print_model_comparison_summary()`

- R3D, MC3, Swin3D, TimeSformer, ViViT 각 모델의 **Top-1 라벨 + 확률**을 한 표로 출력

#### ② 섹션 6 실행 코드 (Cell 22)

```python
print_model_comparison_summary(all_results)
print("결과 이미지 저장:", OUTPUT_DIR.resolve())
```

### 실습 결론

- archery 샘플에 대해 대부분 모델이 **"archery(양궁)"** 를 Top-1으로 예측한다.
- Classical(광학 흐름)부터 딥러닝(3D CNN, Transformer)까지 **서로 다른 방식**이지만, 같은 행동 클립을 **일관되게 인식**함을 확인할 수 있다.

---

## 부록: 출력 파일 위치

| 폴더 | 저장 파일 | 내용 |
|------|-----------|------|
| `output/spatiotemporal/` | `spatial_vs_temporal.png` | 공간 vs 시간 시각화 |
| | `motion_energy_timeline.png` | Motion Energy 그래프 |
| `output/optical_flow/` | `flow_compare_0010.png` | Farneback vs LK 비교 |
| `output/slowfast/` | `slowfast_pathways.png` | Slow/Fast 경로 모자이크 |

---

## 부록: 강의 시 자주 나오는 질문 (FAQ)

**Q. 왜 BGR과 RGB를 변환하나요?**  
A. OpenCV는 BGR 순서로 읽고, PyTorch/Hugging Face 모델은 RGB를 기대합니다. `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`로 변환합니다.

**Q. 왜 프레임을 균등 샘플링하나요?**  
A. 모델마다 고정 프레임 수(8, 16, 32)를 요구하기 때문입니다. 영상 길이와 상관없이 일정 간격으로 뽑아 **클립 길이를 맞춥니다.**

**Q. Optical Flow와 3D CNN의 차이는?**  
A. Optical Flow는 **규칙 기반**으로 "픽셀이 어디로 갔는지" 계산합니다. 3D CNN은 **데이터로부터** "어떤 행동인지"를 학습합니다.

**Q. TimeSformer와 ViViT 중 뭐가 더 좋나요?**  
A. "더 좋다"기보다 **Attention 방식**이 다릅니다. TimeSformer는 공간→시간 순차 분리, ViViT는 Tubelet 단위 3D 토큰을 사용합니다. 입력 프레임 수도 8 vs 32로 다릅니다.

**Q. Swin3D는 CNN인가요 Transformer인가요?**  
A. **Transformer 계열**입니다. 3D Window Attention을 사용하며, 섹션 4(3D CNN)와 섹션 5(Video Transformer)의 경계에 있는 모델입니다.

---

*본 문서는 `01_Temporal_Vision.ipynb` 강의용으로 작성되었습니다.*
