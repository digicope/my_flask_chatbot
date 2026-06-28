# 07 Video Keypoint Detection — 강의용 노트

> **대상:** Video AI 초보자  
> **원본:** `07_Video_Keypoint_Detection.ipynb`  
> **실습 데이터:** `IPAD_sample.zip` (선택) + 포즈 샘플 이미지

---

## 강의 전체 흐름

| Part | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1 | IPAD · 키포인트 개념 | feature vs pose 키포인트 |
| 2 | Harris · ORB · RANSAC | 특징점 검출·매칭 |
| 3 | Keypoint R-CNN | COCO 17 관절 포즈 |
| 4 | 적응형 추적 | LK · z-score · 밀도맵 · manifest |
| 5 | 포즈 시간 안정성 | 관절 각도 변화 |

---

## 키포인트란?

```
프레임
  ↓ 특징점 (ORB/Harris) — 코너·텍스처 추적
  ↓ Pose Keypoint — 어깨·팔꿈치·무릎 17점
  ↓ 시간축 — displacement · 자세 변화
```

02 Motion · 04 이상 탐지와 연결: **displacement** 크면 동작 변화 신호.

---

## Part 1 — IPAD · 키포인트 개념

### 핵심 용어

- **Keypoint(키포인트):** 의미 있는 좌표 (x, y)
- **Feature Keypoint(피처 키포인트):** 코너·돌출부 — **설비 부품** 추적
- **Pose Keypoint(포즈 키포인트):** 사람 **관절** 17점 (COCO)
- **Descriptor(디스크립터):** 키포인트 주변 **패턴 지문** — ORB 매칭용
- **Tracking(트래킹):** 연속 프레임에서 **같은 점** 따라가기
- **Displacement(디스플레이스먼트):** 추적 점 **평균 이동 거리**

| 종류 | 용도 |
|------|------|
| Feature KP | ORB/Harris — 설비 motion |
| Pose KP | Keypoint R-CNN — 작업자 자세 |

### 소스코드

- `load_ipad_clip_frames()` — zip 없으면 checkerboard 합성
- `ensure_pose_sample_image()` — zidane.jpg 등
- 저장: `01_input_samples.png`

---

## Part 2 — 특징점 검출 (Harris · ORB · RANSAC)

### 핵심 용어

- **Harris Corner(해리스 코너):** 밝기 변화 큰 **코너** — `cornerHarris`
- **Shi-Tomasi(시-토마시):** `goodFeaturesToTrack` 코너 — Harris보다 안정
- **ORB(오알비):** Oriented FAST + Rotated BRIEF. 회전·스케일 강함 + **descriptor**
- **Lowe Ratio Test(로우 레이시오):** BFMatcher knn — 좋은 매칭만 선택
- **RANSAC(랜섹):** Random Sample Consensus — outlier 제거 homography
- **Homography(호모그래피):** 평면 간 **透射 변환** 3×3 행렬 H
- **Inlier(인lier):** RANSAC에서 **일관된** 매칭 쌍

### 소스코드

| 함수 | 역할 |
|------|------|
| `compute_harris_response_map()` | 코너 heatmap |
| `detect_harris_corners()` | Harris 좌표 |
| `detect_shi_tomasi_corners()` | Shi-Tomasi |
| `detect_orb_keypoints()` | ORB + descriptor |
| `match_orb_descriptors()` | Lowe ratio |
| `estimate_homography_ransac()` | inlier ratio |

- 저장: `02_feature_keypoints.png`, `02b_detector_compare.png`

> inlier **90%↑** → 두 프레임 거의 같은 장면

---

## Part 3 — 인체 포즈 (Keypoint R-CNN)

### 핵심 용어

- **Keypoint R-CNN(키포인트 알알씨엔):** torchvision COCO pretrained
- **COCO Keypoints(코코 키포인트):** nose, shoulder, elbow, knee … **17점**
- **Skeleton(스켈레톤):** 관절을 **뼈대 선**으로 연결
- **Visibility(비저빌리티):** 관절이 보이는지 0~1
- **Joint Angle(조인트 앵글):** 팔꿈치·무릎·어깨 **각도(도)** — 인체공학

### COCO 17점 (대표)

nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles

### 소스코드

- `get_pose_model()` — `keypointrcnn_resnet50_fpn`
- `detect_human_pose()`, `compute_pose_angles()`
- `render_pose_overlay()`, `draw_pose_skeleton()`
- 저장: `03_human_pose.png`, `03b_pose_angles.png`

> IPAD 설비 영상에 **인물 없으면 0명 정상**

---

## Part 4 — 적응형 추적 · manifest

### 핵심 용어

- **Lucas-Kanade(루카스-카나데):** `calcOpticalFlowPyrLK` — 특징점 추적
- **Adaptive Tracking(어댑티브 트래킹):** 추적 점 부족 시 ORB **재검출**
- **Motion Z-score(모션 지-스코어):** displacement baseline 대비 편차
- **Density Heatmap(밀도 히트맵):** 키포인트 **누적 밀도** — 설비 고정부 후보
- **Manifest(매니페스트):** `keypoint_manifest.json` — 08 통합 입력

### 소스코드

| 함수 | 역할 |
|------|------|
| `track_keypoints_lk()` | LK + ORB reinit |
| `compute_keypoint_displacement()` | 평균 이동 px |
| `compute_displacement_zscore()` | z-score 시계열 |
| `build_keypoint_density_map()` | Gaussian 누적 |
| `run_keypoint_pipeline()` | 전체 → manifest |
| `export_keypoint_manifest()` | JSON v1.1 |

- `motion_spike_frames` — z ≥ 2.0 프레임
- 저장: `04_motion_dashboard.png`, `04_tracking_density.png`

---

## Part 5 — 포즈 시간 안정성

### 핵심 용어

- **Shoulder Center Displacement(숄더 센터):** 좌·우 어깨 중점 이동
- **Angle Delta(앵글 델타):** 프레임 간 **관절 각도 변화**
- **Synthetic Sequence(신테틱):** IPAD에 인물 없을 때 pose 샘플 **미세 이동·회전** 합성

### 소스코드

- `analyze_pose_temporal_stability()`
- `build_pose_demo_sequence()` — synthetic fallback
- 저장: `05_pose_temporal.png`

---

## 부록: 출력 파일

| 파일 | 내용 |
|------|------|
| `output_keypoint/keypoint_manifest.json` | displacement · z-score · spike |
| `figures/02~05_*.png` | 검출·추적·포즈·대시보드 |

---

## 시리즈 연계

- 02 Motion Energy · 04 z-score · **08 unified manifest**

---

*본 문서는 `07_Video_Keypoint_Detection.ipynb` 강의용으로 작성되었습니다.*
