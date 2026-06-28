# 02 Industrial Temporal Pattern — 강의용 노트

> **대상:** Video AI를 처음 접하는 초보자 (01 Temporal Vision 수강 후)  
> **원본:** `02_Industrial_Temporal_Pattern.ipynb`  
> **실습 데이터:** `IPAD_sample.zip` → `IPAD_Sample/` (R01 설비, 클립 01, 약 11MB)

---

## 강의 전체 흐름

| 섹션 | 주제 | 한 줄 요약 |
|------|------|-----------|
| 0 | 환경 및 IPAD 데이터 | 산업 설비 프레임 이미지를 읽고 분석 준비 |
| 1 | IPAD(아이패드) 데이터셋 탐색 | 설비·클립 구조를 이해하고 train/test 로드 |
| 2 | Motion Energy(모션 에너지) | 프레임 차분으로 설비 동작 강도 측정 |
| 3 | 주기(Period) 추정 | Autocorrelation·FFT로 1사이클 길이 찾기 |
| 4 | 사이클(Cycle) 분할 | 주기 단위로 시계열·영상 나누기 |
| 5 | Optical Flow(광학 흐름) | 설비 부품의 이동 방향·속도 시각화 |
| 6 | 정상 vs 이상 | Z-score로 test 사이클 편차 탐지 |
| 7 | STFT(에스티에프티) | 시간-주파수 스펙트로그램으로 국소 패턴 분석 |
| 8 | DTW(디티더블유) | 사이클 곡선 형태 정렬·이상 거리 |
| 9 | CUSUM(큐섬) & Drift | change-point·주기 변화 시점 탐지 |
| 10 | Mahalanobis(마할라노비스) | 다변량 feature 공간 이상 거리 |
| 11 | Operational Phase(운전 페이즈) | 사이클 내 4단계 운전 상태 분할 |

---

## 0. 환경 및 데이터 준비

### 이 섹션에서 배우는 것

- **IPAD(Industrial Process Anomaly Detection, 아이패드)** 산업 설비 이상 탐지 데이터셋의 구조를 이해한다.
- `IPAD_sample.zip` 자동 압축 해제, 프레임 로드, 결과 저장 경로를 설정한다.

### 핵심 용어

- **IPAD(아이패드, Industrial Process Anomaly Detection):** 산업 공정(설비) 영상에서 **이상(Anomaly)** 을 탐지하기 위한 공개 데이터셋. Liu et al., arXiv:2404.15033
- **VAD(브이에이디, Video Anomaly Detection):** 비디오 이상 탐지. 정상 패턴과 다른 구간을 찾는 기술
- **Industrial VAD(인더스트리얼 브이에이디):** 교통·보행 VAD와 달리, **설비가 주기적으로 반복 동작**한다는 점이 핵심
- **Periodicity(피리어디시티, 주기성):** 같은 동작이 일정 간격으로 반복되는 성질. 산업 설비 VAD의 가장 중요한 Temporal 특징
- **Temporal Pattern(템포럴 패턴, 시간 패턴):** 시간에 따라 반복·변화하는 동작 패턴
- **Frame Sequence(프레임 시퀀스):** `000.jpg`, `001.jpg`, … 순서대로 나열된 이미지 묶음 (mp4가 아님)
- **Training / Testing(트레이닝 / 테스팅):** IPAD에서 training=정상 위주, testing=이상 구간 포함 가능

### IPAD 데이터 구조

```
IPAD_Sample/
  └── R01/                    ← 설비 코드 (Device)
        ├── training/
        │     └── frames/
        │           └── 01/   ← 클립 ID
        │                 ├── 000.jpg
        │                 ├── 001.jpg
        │                 └── ...
        └── testing/
              └── frames/
                    └── 01/
                          └── *.jpg
```

### 필요 패키지 (01번 + 추가)

```
opencv-python, numpy, matplotlib, scipy
```

- **SciPy(사이파이):** Autocorrelation, FFT, STFT, find_peaks 등 신호 처리 함수 제공

### 소스코드 핵심 기능

#### ① 전역 설정 (Cell 3)

| 변수 | 역할 |
|------|------|
| `IPAD_ZIP_NAMES` | `IPAD_sample.zip` / `IPAD_Sample.zip` 파일명 후보 |
| `IPAD_EXTRACT_DIR` | 압축 해제 폴더 `IPAD_Sample/` |
| `OUTPUT_DIR` / `FIG_DIR` | 결과 저장 `output_ipad/figures/` |
| `SELECTED_DEVICE = "R01"` | 분석할 설비 고정 |
| `SELECTED_CLIP = "01"` | training 클립 ID |
| `SELECTED_TEST_CLIP = "01"` | testing 클립 ID |

#### ② `IpadClip` dataclass

- 설비(`device`), split(`training`/`testing`), 클립 ID, 프레임 경로 목록을 하나로 묶는 데이터 클래스
- `num_frames`: 프레임 수 (Motion Energy 길이 = num_frames − 1)

#### ③ IPAD 데이터 로드 함수군

| 함수 | 역할 |
|------|------|
| `find_ipad_zip()` | 현재 폴더에서 zip 파일 탐색 |
| `ensure_ipad_sample()` | zip → `IPAD_Sample/` 자동 압축 해제 (이미 있으면 스킵) |
| `find_ipad_root()` | `{설비}/training/frames/` 구조가 있는 루트 찾기 |
| `list_ipad_devices()` | 설비 목록 (R01, S03 등) |
| `list_clips()` | training/testing 클립 ID 목록 |
| `load_ipad_clip()` | jpg/png 프레임 경로 수집 → `IpadClip` 반환 |
| `read_frames_bgr()` | 각 jpg를 OpenCV BGR 배열로 읽기 |
| `resolve_ipad_session()` | zip 해제 → train 클립 + test 클립 한 번에 결정 |
| `try_load_anomaly_labels()` | `frame_labels_{device}.npy` 이상 라벨 로드 (0=정상, 1=이상) |

> **강의 포인트:** IPAD는 mp4가 아니라 **프레임 이미지 폴더**입니다. OpenCV `VideoCapture` 대신 `cv2.imread`로 한 장씩 읽습니다.

---

## 1. IPAD 데이터셋 탐색

### 이 섹션에서 배우는 것

- 데이터셋이 정상적으로 준비되었는지 확인하고, train 클립의 해상도·프레임 수·재생 시간을 파악한다.

### 핵심 용어

- **Device(디바이스, 설비):** IPAD에서 R01, S03 등 공장 설비 종류를 구분하는 코드
- **Clip(클립):** 한 설비의 연속 촬영 영상 1개 (프레임 폴더 1개)
- **Split(스플릿):** `training`(학습·정상) / `testing`(테스트·이상 포함) 구분
- **pathlib(패스리브):** Python 경로 처리 라이브러리 (`Path`)
- **zipfile(집파일):** Python zip 압축 해제 모듈

### 소스코드 핵심 기능 (Cell 6 실행)

```python
ipad_root, train_clip, test_clip = resolve_ipad_session()
train_frames = read_frames_bgr(train_clip)
fps_assumed = 30.0  # IPAD 논문: 대부분 30 FPS
duration = len(train_frames) / fps_assumed
```

- 설비 목록, training/testing 클립 목록 출력
- 해상도, 프레임 수, 추정 재생 시간(초) 출력

> **강의 포인트:** `fps_assumed = 30.0`은 IPAD 논문 기준 가정값입니다. 프레임 수 ÷ FPS = 재생 시간(초).

---

## 2. Motion Energy — 설비 동작 강도

### 이 섹션에서 배우는 것

- 연속 프레임 차분으로 **프레임마다 하나의 숫자**(움직임 강도)를 만든다.
- 산업 설비의 **동작 peak / 정지 trough** 패턴을 시계열로 확인한다.

### 핵심 용어

- **Motion Energy(모션 에너지):** `|F(t+1) − F(t)|` 픽셀 평균. 값이 크면 움직임, 작으면 정지
- **Time Series(타임 시리즈, 시계열):** 시간 순서대로 나열된 숫자 데이터
- **Peak(피크):** 시계열에서 값이 **최대**인 지점 → 설비가 가장 활발히 움직이는 순간
- **Trough(트로프):** 시계열에서 값이 **최소**인 지점 → 정지·대기 구간
- **Grayscale(그레이스케일):** 흑백 변환. 색 정보 없이 밝기만 사용 → 계산 빠름
- **Farneback(파르너백):** Dense Optical Flow 알고리즘. 01번 노트북과 동일
- **Magnitude(매그니튜드, 크기):** 이동 벡터의 길이 = 움직임 속도

### Motion Energy vs Flow Magnitude

| 지표 | 계산 | 의미 |
|------|------|------|
| Motion Energy | `absdiff` 픽셀 평균 | "변화량" — 픽셀이 얼마나 달라졌는가 |
| Flow Magnitude | Farneback 이동 벡터 크기 평균 | "이동량" — 픽셀이 얼마나, 어느 방향으로 움직였는가 |

> **핵심:** 산업 설비는 peak/trough가 **주기적으로 반복** → 이후 주기 추정·사이클 분할의 **1차 입력 신호**

### 소스코드 핵심 기능

#### ① `compute_motion_energy(frames)`

```python
for i in range(len(frames) - 1):
    g0, g1 = 그레이스케일 변환
    energies.append(mean(absdiff(g0, g1)))
```

#### ② `compute_flow_magnitude_mean(frames)`

- Farneback → `cartToPolar` → magnitude 공간 평균

#### ③ `plot_motion_signals(motion, flow_mag, ...)`

- 상단: Motion Energy vs 시간(초)
- 하단: Flow Magnitude vs 시간(초)
- 저장: `01_motion_energy_train.png`

---

## 3. 주기(Period) 추정 — Autocorrelation & FFT

### 이 섹션에서 배우는 것

- Motion Energy 시계열에서 **1사이클 = 몇 프레임**인지 추정한다.
- Autocorrelation(자기상관)과 FFT(푸리에 변환) **두 방법**으로 교차 검증한다.

### 핵심 용어

- **Period(피리어드, 주기):** 같은 패턴이 반복되는 간격. IPAD 설비 1사이클은 보통 **5~30초**
- **Cycle(사이클):** 주기 1회분의 동작 (예: 리프트 1회 상승·하강)
- **Autocorrelation(오토코릴레이션, 자기상관):** 시계열을 자기 자신과 lag(지연)만큼 밀어 비교. lag=T에서 peak → T 프레임마다 반복
- **Lag(래그, 지연):** 두 시점 사이의 프레임 간격
- **FFT(에프에프티, Fast Fourier Transform, 고속 푸리에 변환):** 시계열을 주파수 영역으로 변환. 가장 강한 주파수 f → 주기 = 1/f
- **Dominant Frequency(도미넌트 프리퀀시, 지배 주파수):** 스펙트럼에서 가장 강한 주파수 성분
- **DC Component(디씨 컴포넌트):** 0 Hz 성분 (평균 수준). 주기 추정 시 제외
- **Hz(헤르츠):** 초당 반복 횟수. 0.1 Hz = 10초에 1번 반복

### Autocorrelation vs FFT

| 방법 | 원리 | 결과 |
|------|------|------|
| Autocorrelation | lag=T에서 상관 peak | 주기 = T **프레임** |
| FFT | dominant frequency f | 주기 = 1/f **초** |

> **강의 포인트:** 두 값이 **대략 일치**하면 주기 추정 신뢰도가 높습니다.

### 소스코드 핵심 기능

#### ① `estimate_period_autocorr(series, min_lag=5)`

1. DC 제거 (`series - mean`)
2. `scipy.signal.correlate`로 자기상관 계산
3. lag=0 근처 trivial peak 제외 (`min_lag` 이후)
4. 최대 peak lag → **estimated_period_frames**

#### ② `estimate_period_fft(series, fps=30.0)`

1. `scipy.fft.rfft` / `rfftfreq`
2. DC(0 Hz) 제외 후 최대 스펙트럼 bin
3. `period_sec = 1 / dominant_freq`

#### ③ `plot_period_analysis(...)`

- 상단: Motion Energy + 추정 주기마다 수직 점선
- 하단: Autocorrelation 곡선 + peak lag 표시
- 저장: `02_period_analysis.png`

---

## 4. 주기 단위 사이클 분할

### 이 섹션에서 배우는 것

- 추정된 주기 `period`로 Motion Energy를 **사이클 0, 1, 2, …** 로 잘라 비교한다.
- 정상 설비면 사이클 곡선이 **서로 잘 맞물리고**, 이상이 있으면 특정 사이클만 **어긋난다**.

### 핵심 용어

- **Segmentation(세그멘테이션, 분할):** 긴 시계열·영상을 의미 있는 구간으로 나누는 것
- **Overlay(오버레이, 겹치기):** 여러 사이클 곡선을 같은 시간축(0~1사이클)에 겹쳐 그림
- **Keyframe(키프레임):** 대표 프레임. 시작·1/4·1/2·3/4 지점
- **Period Memory(피리어드 메모리):** IPAD 논문 개념 — 한 주기 단위로 패턴을 기억·비교
- **Sliding Window(슬라이딩 윈도우):** 고정 길이 창을 시간축上 이동하며 분석

### 소스코드 핵심 기능

#### ① `segment_cycles(motion, period)`

```python
n_cycles = len(motion) // period
return [motion[i*period : (i+1)*period] for i in range(n_cycles)]
```

- 마지막 `period` 미만 구간은 버림

#### ② `plot_cycle_overlay(cycles, period, fps)`

- 최대 8개 사이클 곡선을 0~1사이클 시간축에 겹쳐 표시
- 저장: `03_cycle_overlay.png`

#### ③ `visualize_cycle_keyframes(frames, period, cycle_index=0)`

- 사이클 시작·1/4·1/2·3/4 프레임 4장 나란히 표시
- 저장: `04_cycle_keyframes.png`

---

## 5. Optical Flow — 설비 부품 이동

### 이 섹션에서 배우는 것

- Motion Energy **peak 구간**에서 Dense Optical Flow를 계산한다.
- HSV 컬러맵으로 **이동 방향(색)** 과 **속력(밝기)** 을 한 장에 표현한다.

### 핵심 용어

- **Optical Flow(옵티컬 플로우, 광학 흐름):** 연속 프레임에서 픽셀 이동 벡터 (dx, dy) 추정
- **Farneback(파르너백):** Dense Optical Flow — **모든 픽셀**의 이동량 계산
- **Dense(덴스):** 모든 픽셀 대상 (Sparse의 반대)
- **HSV(에이치에스브이):** Hue=방향, Value=속력으로 flow 시각화
- **Pseudo-color(슈도컬러, 의사색):** 흑백 flow를 색상으로 변환해 보기 쉽게 표현
- **ROI(알오아이, Region of Interest):** 관심 영역. 특정 부품만 움직임 이상 가능

### 왜 peak 구간인가?

Motion Energy가 **최대**인 순간 = 설비가 가장 활발히 움직일 때 → flow 패턴이 **가장 뚜렷**

### 소스코드 핵심 기능

#### ① `flow_to_hsv(flow)`

- `cartToPolar` → angle→Hue, magnitude→Value → BGR 이미지

#### ② `visualize_peak_motion_flow(frames, motion)`

1. `peak_idx = argmax(motion)` — Motion Energy 최대 프레임
2. `calcOpticalFlowFarneback(frames[peak_idx], frames[peak_idx+1])`
3. 3열: 이전 프레임 | 다음 프레임 | Flow HSV
4. 저장: `05_optical_flow_peak.png`

---

## 6. 정상 vs 이상 — 시간 패턴 편차

### 이 섹션에서 배우는 것

- train(정상) 사이클의 Motion Energy 분포를 **baseline(기준선)** 으로 삼는다.
- test 사이클이 baseline에서 **통계적으로 벗어나면** 이상 후보로 표시한다.

### 핵심 용어

- **Baseline(베이스라인, 기준선):** 정상(train) 데이터로 계산한 평균·표준편차
- **Z-score(지-스코어):** `(x − μ) / σ`. 값이 얼마나 평균에서 벗어났는지 표준편차 단위로 표현
- **μ(뮤, mu):** 평균 (mean)
- **σ(시그마, sigma):** 표준편차 (standard deviation)
- **Anomaly(어노멀리, 이상):** 정상 패턴과 다른 구간·사이클
- **Reconstruction Error(리컨스트럭션 에러, 복원 오차):** 딥러닝 VAD에서 정상 모델이 test를 복원할 때의 오차. Z-score와 직관적으로 대응
- **Period Classification(피리어드 클래시피케이션):** IPAD 논문 VAD의 핵심 — 주기 패턴 분류

### Z-score 이상 판정

```
1. train 사이클별 평균 Motion Energy → μ, σ 계산
2. test 각 사이클: z = (x − μ) / σ
3. |z| > 2 → train에서 통계적으로 드문 패턴 → 이상 후보 (빨간 bar)
```

> **강의 포인트:** 규칙 기반이지만, 딥러닝 VAD의 "정상 분포 대비 deviation" 개념과 **직관적으로 대응**합니다.

### 소스코드 핵심 기능

#### ① `cycle_energy_profile(motion, period)`

- 각 사이클 Motion Energy **평균** 1개 → 사이클 단위 scalar 시계열

#### ② `compare_train_test_patterns(train_motion, test_motion, period)`

- 상단: train 사이클별 에너지 bar + mean 점선
- 하단: test 사이클별 bar, |z|>2 빨간색, ±2σ 점선
- 저장: `06_motion_energy_test.png`, `07_train_test_cycle_compare.png`

---

## 7. STFT 스펙트로그램 — 시간-주파수 패턴

### 이 섹션에서 배우는 것

- Motion Energy를 **짧은 윈도우**마다 FFT → "그 순간 어떤 주파수가 강한가" 2D 히트맵
- **전역 주기**(Autocorr) vs **국소 주기 변화**(STFT) 차이를 이해한다.

### 핵심 용어

- **STFT(에스티에프티, Short-Time Fourier Transform, 단시간 푸리에 변환):** 긴 시계열을 짧은 구간으로 잘라 각각 FFT. 시간×주파수 2D 표현
- **Spectrogram(스펙트로그램):** STFT 결과를 색상 히트맵으로 표현한 것
- **Window(윈도우):** STFT에서 한 번에 분석하는 구간 길이 (`nperseg`)
- **Overlap(오버랩):** 인접 윈도우 간 겹침 비율 (`noverlap`). 75% overlap → 시간축 매끄럽게
- **Drift(드리프트, 표류):** 주기가 서서히 변하는 현상 (사이클 길이 연장/단축)
- **Transient(트랜지언트, 과도):** 일시적 변화. stripe가 흐릿·사라지는 구간

### Autocorr/FFT vs STFT

| 방법 | 보는 것 | 범위 |
|------|---------|------|
| Autocorr / FFT | 대표 주기 | **전역(global)** |
| STFT | 시간 구간별 주파수 | **국소(local)** |

### 스펙트로그램 읽는 법

- **수평 stripe 안정** → 주기가 **일정** (정상)
- **stripe 위치 변화** → **주기 drift** (이상 단서)
- **특정 시간 stripe 흐릿** → **과도(transient)** 또는 동작 중단

### 소스코드 핵심 기능

#### ① `plot_motion_stft(motion, fps=30.0, ...)`

1. `scipy.signal.stft(x, fs=fps, nperseg=..., noverlap=75%)`
2. power → dB 스케일 (`10*log10`)
3. 상단: 원 Motion Energy / 하단: STFT 히트맵 (magma colormap)
4. cyan 점선 = Autocorr 전역 주기 주파수 참조
5. 저장: `08_stft_train.png`, `09_stft_test.png`

---

## 8. DTW — Dynamic Time Warping(다이나믹 타임 워핑)

### 이 섹션에서 배우는 것

- 두 시계열의 **길이·속도가 달라도** 비선형으로 시간축을 맞춰 유사도를 측정한다.
- Z-score(평균 1개)보다 **사이클 전체 곡선 shape**를 비교하는 고급 이상 탐지.

### 핵심 용어

- **DTW(디티더블유, Dynamic Time Warping, 동적 시간 정렬):** 두 시계열을 비선형 warp(늘리기·줄이기)로 정렬해 최소 누적 비용(dist) 계산
- **Warping(워핑, 정렬):** 시간축을 비선형으로 늘리거나 줄여 두 곡선을 맞추는 것
- **Template(템플릿, 기준 곡선):** 정상 train 사이클들의 median → outlier에 강한 기준
- **Dynamic Programming(다이나믹 프로그래밍, DP):** `d[i,j]` 테이블로 insert/delete/match 최소 경로 탐색
- **Robust(로버스트, 견고한):** peak가 몇 프레임 밀려도 DTW는 warp로 흡수 — L2 거리보다 산업 주기에 적합

### Z-score vs DTW

| 방법 | 비교 대상 |
|------|-----------|
| Z-score (섹션 6) | 사이클 **평균 에너지** 1개 숫자 |
| DTW (섹션 8) | 사이클 **전체 곡선 shape** |

### 소스코드 핵심 기능

#### ① `dtw_distance(a, b)`

- O(n·m) DP 테이블. insert/delete/match 중 최소 누적 비용

#### ② `normalize_cycle(cycle)`

- z-score 정규화 → 진폭·오프셋 차이 제거

#### ③ `build_cycle_template(cycles)`

- 정상 사이클 z-score 후 **element-wise median** → robust template

#### ④ `dtw_anomaly_scores(cycles, template)`

- 각 사이클 ↔ template DTW 거리 배열

#### ⑤ `plot_dtw_analysis(...)`

- 상단: template + train/test 사이클 곡선 overlay
- 하단: train/test DTW distance bar, |z|>2 이상 후보
- 저장: `10_dtw_cycle_analysis.png`

---

## 9. CUSUM & 주기 Drift — 순차 이상·국소 주기 변화

### 이 섹션에서 배우는 것

- **CUSUM**으로 "언제부터 패턴이 달라졌는지" **프레임 단위 change-point** 탐지
- **Rolling Period**로 시간에 따른 **국소 주기 변화** 추적

### 핵심 용어

- **CUSUM(큐섬, Cumulative Sum, 누적합):** baseline에서 한쪽으로 **지속적으로** 벗어나면 누적합 S+, S−가 threshold 초과 → alarm
- **Change-point(체인지포인트, 변화점):** 시계열에서 패턴이 바뀌는 시점
- **Drift(드리프트):** CUSUM에서 작은 흔들림 무시용 (baseline의 2%)
- **Threshold(쓰레숄old, 임계값):** alarm 기준 (코드: 5.0)
- **Rolling Window(롤링 윈도우, 슬라이딩 윈도우):** 고정 길이 구간을 이동하며 Autocorr peak → **국소 주기** 시계열
- **Global Period(글로벌 피리어드, 전역 주기):** 영상 전체 Autocorr로 구한 대표 주기
- **Local Period(로컬 피리어드, 국소 주기):** 윈도우마다 추정한 주기. 전역에서 벗어나면 drift 이상

### 3단 subplot 구성

1. Motion Energy + CUSUM alarm 수직선 (빨간)
2. S+, S− 누적곡선 + threshold
3. Rolling Autocorr 국소 주기 vs 전역 주기 (녹색 점선)

### 소스코드 핵심 기능

#### ① `cusum_detect(series, target, drift, threshold=5.0)`

- S+ = max(0, S+ + val − drift) — baseline보다 지속적으로 큰 편차
- S− = min(0, S− + val + drift) — baseline보다 지속적으로 작은 편차
- threshold 초과 → alarm_indices

#### ② `rolling_period_estimate(motion, window, min_lag, step)`

- 슬라이딩 윈도우 Autocorr peak → (centers, local_periods)

#### ③ `plot_cusum_and_drift(...)`

- window ≈ 3×global_period (국소 주기 추정에 충분한 길이)
- 저장: `11_cusum_drift_train.png`, `12_cusum_drift_test.png`

---

## 10. Mahalanobis — 다변량 사이클 feature 이상

### 이 섹션에서 배우는 것

- 한 사이클을 **9차원 feature vector**로 요약한다.
- train feature 분포 대비 test 사이클의 **Mahalanobis 거리**로 다변량 이상 탐지.

### 핵심 용어

- **Feature Vector(피처 벡터, 특징 벡터):** 한 사이클을 숫자 여러 개로 요약한 배열
- **Multivariate(멀티베리에이트, 다변량):** 여러 feature를 동시에 고려
- **Mahalanobis Distance(마할라노비스 디스턴스, 마할라노비스 거리):** 평균·**공분산(covariance)** 을 반영한 통계적 거리. feature 간 **상관**까지 고려
- **Covariance(코베리언스, 공분산):** 두 feature가 함께 변하는 정도
- **PCA(피씨에이, Principal Component Analysis, 주성분 분석):** 고차원 feature를 2D에 투영해 scatter plot
- **PC1 / PC2(피씨원 / 피씨투):** 1·2번째 주성분 축
- **SVD(에스브이디, Singular Value Decomposition):** PCA에 사용하는 행렬 분해 (sklearn 불필요)
- **Spectral Centroid(스펙트럴 센트로이드, 스펙트럼 중심):** 주파수 에너지가 몰린 대역 — 곡선 "빠르기" 느낌
- **AUC(에이유씨, Area Under Curve):** 곡선 적분 — 총 움직임량

### 사이클 feature 9차원

| feature | 의미 |
|---------|------|
| mean_E | 평균 동작 강도 |
| std_E | 사이클 내 변동성 |
| max_E | peak 강도 |
| peak_pos | peak 위치 (0~1) |
| spec_centroid | 스펙트럼 중심 |
| half_ratio | 전반/후반 에너지 비 |
| AUC | 곡선 적분 — 총 움직임량 |
| flow_mean | Optical Flow 평균 |
| flow_std | Optical Flow 표준편차 |

### 소스코드 핵심 기능

#### ① `extract_cycle_features(cycle, flow_cycle)`

- Motion Energy + (선택) Flow 통계 → 7~9차원 numpy 배열

#### ② `build_feature_matrix(motion, period, flow)`

- 전체 motion → `(n_cycles × n_features)` 행렬

#### ③ `mahalanobis_scores(train_feats, test_feats, reg=1e-3)`

- train 평균 μ, 공분산 Σ (+ regularization)
- 거리 = √((x−μ)ᵀ Σ⁻¹ (x−μ))

#### ④ `plot_mahalanobis_analysis(...)`

- 좌: PCA 2D scatter (train=녹색, test=파랑)
- 우: Mahalanobis distance bar, +2σ 초과 이상 후보
- 저장: `13_mahalanobis.png`

---

## 11. Operational Phase — 사이클 내 상태 전이

### 이 섹션에서 배우는 것

- 한 사이클 Motion Energy에서 peak를 찾아 **4단계 운전 Phase**로 나눈다.
- train vs test **phase별 에너지 heatmap**으로 "어느 동작 단계에서 이상인지" 확인.

### 핵심 용어

- **Operational Phase(오퍼레이셔널 페이즈, 운전 단계):** 사이클 내 설비 상태 구간 (대기→가속→작업→복귀)
- **find_peaks(파인드 피크스):** SciPy 함수. 시계열에서 peak(극대값) 위치 검출
- **Prominence(프로미넌스, 돌출도):** 주변 대비 peak 높이. 약한 노이즈 peak 무시
- **Distance(디스턴스):** peak 간 최소 간격. 너무 촘촘한 peak 제거
- **Heatmap(히트맵):** phase별 mean energy를 색상 행렬로 표현
- **State Transition(스테이트 트랜지션, 상태 전이):** Idle → Accel → Peak → Decel 순서 변화

### 4단계 Phase

| Phase | 의미 (산업 설비 예) |
|-------|---------------------|
| Idle/Start(아이들/스타트) | 대기·시작 |
| Accel(액셀, Acceleration) | 가속·기동 |
| Peak/Work(피크/워크) | 최대 부하·작업 |
| Decel/Return(디셀/리턴) | 감속·복귀 |

### 소스코드 핵심 기능

#### ① `segment_operational_phases(cycle_motion, n_phases=4)`

1. `find_peaks(c, distance=..., prominence=...)`
2. peak 기준 boundary 설정 → 4구간 segments
3. 프레임별 phase label (0~3) 반환

#### ② `plot_phase_segmentation(...)`

- 상단: Motion Energy + phase별 색 영역
- 중단: phase label timeline (imshow)
- 하단: phase마다 대표 키프레임 1장
- 저장: `14_phase_segmentation.png`

#### ③ `compare_phase_energy_profiles(train_cycles, test_cycles)`

- train vs test phase별 mean energy heatmap
- 특정 phase(예: Peak/Work)만 다르면 **그 단계에서 이상**
- 저장: `15_phase_heatmap.png`

---

## 01 Temporal Vision과의 관계

| | 01 Temporal Vision | 02 Industrial Temporal Pattern |
|---|-------------------|-------------------------------|
| **목적** | 일반 Video AI (행동 인식) | 산업 설비 **주기·이상** 분석 |
| **데이터** | archery.mp4 (양궁) | IPAD 프레임 시퀀스 (설비) |
| **공통 기법** | Motion Energy, Farneback Optical Flow | 동일 |
| **01만** | 3D CNN, TimeSformer, ViViT | — |
| **02만** | — | Autocorr, STFT, DTW, CUSUM, Mahalanobis, Phase |

> **핵심:** 01에서 배운 Motion/Flow를 02에서 **산업 주기 분석·이상 탐지**에 특화해 확장합니다.

---

## 부록: 출력 파일 위치

| 파일 | 내용 |
|------|------|
| `01_motion_energy_train.png` | 정상 Motion Energy·Flow 시계열 |
| `02_period_analysis.png` | 주기 추정 + Autocorrelation |
| `03_cycle_overlay.png` | 사이클 곡선 overlay |
| `04_cycle_keyframes.png` | 사이클 키프레임 4장 |
| `05_optical_flow_peak.png` | peak 구간 Optical Flow HSV |
| `06_motion_energy_test.png` | test Motion Energy |
| `07_train_test_cycle_compare.png` | Z-score train vs test |
| `08_stft_train.png` / `09_stft_test.png` | STFT 스펙트로그램 |
| `10_dtw_cycle_analysis.png` | DTW template·거리 |
| `11_cusum_drift_train.png` / `12_cusum_drift_test.png` | CUSUM·Rolling Period |
| `13_mahalanobis.png` | PCA 2D + Mahalanobis bar |
| `14_phase_segmentation.png` | Phase 분할·키프레임 |
| `15_phase_heatmap.png` | Phase별 energy heatmap |

저장 경로: `output_ipad/figures/`

---

## 부록: 이상 탐지 기법 비교 (섹션 6~11)

| 기법 | 비교 대상 | 잡아내는 이상 유형 |
|------|-----------|-------------------|
| Z-score | 사이클 평균 에너지 1개 | 전체 강도 편차 |
| STFT | 시간-주파수 패턴 | 국소 주기 drift, 과도 |
| DTW | 사이클 곡선 shape | 형태·타이밍 어긋남 |
| CUSUM | 프레임별 에너지 | change-point (언제부터?) |
| Rolling Period | 국소 주기 | 주기 연장/단축 |
| Mahalanobis | 9차원 feature | 다변량 복합 이상 |
| Phase | 사이클 내 4단계 | 특정 운전 단계 이상 |

---

## 부록: 강의 FAQ

**Q. IPAD_sample.zip이 없으면?**  
A. `resolve_ipad_session()`에서 `FileNotFoundError`와 안내 메시지가 출력됩니다. 노트북과 **같은 폴더**에 zip을 배치하세요.

**Q. training과 testing의 차이는?**  
A. training은 **정상** 위주 클립, testing은 **이상 구간 포함** 가능. test를 train baseline과 비교합니다.

**Q. 왜 mp4가 아니라 jpg 폴더인가요?**  
A. IPAD 원본 형식이 프레임 이미지 시퀀스입니다. 프레임 단위 라벨·분석에 유리합니다.

**Q. Z-score |z|>2는 왜 2인가요?**  
A. 정규분포 가정 시 약 95% 구간. **규칙 기반 baseline**이며, 실무에서는 threshold를 데이터에 맞게 조정합니다.

**Q. DTW가 Z-score보다 나은 이유는?**  
A. peak가 몇 프레임 밀려도 warp로 흡수합니다. 산업 설비처럼 **곡선 형태**가 중요할 때 유리합니다.

**Q. IPAD 논문 본 모델과 이 실습의 차이는?**  
A. 논문은 reconstruction + period classification 등 **딥러닝 VAD**. 본 실습은 그 **전 단계**로 Motion Energy·통계 기법을 직접 구현합니다.

---

## 참고 자료

- [IPAD GitHub](https://github.com/LJF1113/IPAD)
- [IPAD Project Page](https://ljf1113.github.io/IPAD_VAD/)
- Liu et al., *IPAD: Industrial Process Anomaly Detection Dataset*, [arXiv:2404.15033](https://arxiv.org/abs/2404.15033)
- [전체 IPAD_dataset.zip (Google Drive)](https://drive.google.com/file/d/1SwSScNzhzE6t8N9JxK843SsqthmFdZIv/view) — 약 8GB

---

*본 문서는 `02_Industrial_Temporal_Pattern.ipynb` 강의용으로 작성되었습니다.*
