# 09 Video Object Tracking & Equipment State — 강의용 노트

> **대상:** Video AI를 처음 접하는 초보자 (02 Motion · 08 Manifest 수강 후)  
> **원본:** `09_Video_Object_Tracking_Equipment_State.ipynb`  
> **실습 데이터:** `IPAD_sample.zip` (선택) 또는 합성 이동 객체 시퀀스

---

## 강의 전체 흐름

| Part | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1 | 영상 수집 · 원본 시각화 | IPAD/decord 로드 · ROI·Motion Energy 확인 |
| 2 | YOLOv8 검출 | Local Detection · motion blob 하이브리드 |
| 3 | ByteTrack 추적 | MOT · ID 유지 |
| 4 | 설비 동작 상태 | ROI · 속도 · motion z → RUNNING/IDLE/ANOMALY |
| 5 | 상황 요약 에이전트 | 한국어 관제 리포트 |
| 6 | ChromaDB 검색 | 시맨틱 관제 질의 |
| 7 | 관제 대시보드 · manifest | MOT 지표 · tracking_manifest.json |

---

## PoC 파이프라인 개요

교재 **8-5 Video AI Service PoC** 아키텍처: **검출 → 추적 → 상황 추론 → 검색**

```
CCTV / IPAD 프레임 시퀀스
    ↓ decord 가속 (또는 OpenCV)
YOLOv8 검출 + ByteTrack ID 유지
    ↓ 구역 이탈 · 속도 · motion z-score
상황 요약 (자연어 리포트)
    ↓ embedding
ChromaDB 시맨틱 검색 + 관제 대시보드
```

02 **Motion Energy** · 07 **키포인트 displacement**와 연결: 추적 궤적의 **속도·구역**이 설비 **가동/정지/이상**을 설명합니다.

---

## 환경 설정

### 핵심 용어

- **HAS_DECORD / HAS_YOLO / HAS_CHROMA:** 패키지 로드 여부. `False`면 **자동 폴백**
- **MAX_FRAMES / FRAME_STRIDE:** 실습 속도용 샘플링 (stride=2 → 0,2,4,…)
- **YOLO_CONF / YOLO_IOU:** 검출 confidence·NMS IoU 임계값
- **EQUIPMENT_ZONES:** 화면 비율(0~1) **ROI** — operation_zone, idle_zone
- **VELOCITY_*_THRESH:** px/frame 기준 가동·정지 구분

### 주요 패키지

| 패키지 | 역할 |
|--------|------|
| `ultralytics` | YOLOv8 + ByteTrack |
| `chromadb` | 상황 요약 벡터 저장·검색 (선택) |
| `decord` | mp4 고속 디코딩 (선택) |

```bash
pip install ultralytics chromadb decord
```

### 소스코드 핵심 (설정)

| 변수 | 역할 |
|------|------|
| `YOLO_MODEL_NAME = "yolov8n.pt"` | nano — 가볍고 빠름 |
| `TRACK_CLASS_NAMES` | person, truck, car 등 COCO 클래스 |
| `MOTION_DIFF_THRESH` | IPAD motion blob 검출 |
| `ZONE_VIOLATION_FRAMES = 2` | 연속 N프레임 이탈 → 이상 |
| `MOTION_Z_ANOMALY = 2.2` | motion z-score 이상 임계 |
| `OUTPUT_DIR` | `output_tracking/` |

---

## Part 1 — 영상 수집 · 원본 시각화

### 이 섹션에서 배우는 것

- IPAD 프레임 시퀀스 또는 **합성 영상**을 로드한다.
- 추적 전 **원본 품질·동작 리듬·ROI**를 눈으로 확인한다.

### 핵심 용어

- **decord(디코드):** 딥러닝용 고속 비디오 디코더. `get_batch`로 임의 프레임 일괄 로드
- **BGR(비지알):** OpenCV 색 순서 (Blue-Green-Red)
- **Synthetic Sequence(신테틱 시퀀스):** IPAD 없을 때 이동 사각형 합성 영상
- **Data Ingestion(데이터 인제션):** CCTV·프레임 시퀀스 수집 단계

### 소스코드 핵심

| 함수 | 역할 |
|------|------|
| `ensure_ipad_sample()` | zip 압축 해제 또는 기존 폴더 사용 |
| `load_ipad_clip_frames()` | testing 우선, stride 샘플링 |
| `read_image_bgr()` | `fromfile` + `imdecode` (한글 경로) |
| `make_synthetic_equipment_sequence()` | 합성 폴백 |
| `plot_source_overview()` | 8프레임 그리드 + ROI + motion 곡선 |

**출력:** `frames_bgr`, `frame_paths`, `source_label` ("IPAD" | "synthetic")  
**저장:** `figures/01_source_overview.png`

> **강의 포인트:** Motion Energy(02번 동일)가 이후 z-score·상태 분류의 **기준 신호**.

---

## Part 2 — YOLOv8 객체 검출 (Local Detection)

### 이 섹션에서 배우는 것

- 프레임마다 **바운딩 박스(bbox)** 후보를 찾는다.
- IPAD 산업 영상은 COCO 객체가 거의 없어 **motion blob**을 주 검출로 사용한다.

### 핵심 용어

- **Detection(디텍션, 검출):** "무엇이 어디에?" — bbox + class + confidence
- **YOLOv8(와이오엘오-v8):** Ultralytics 실시간 객체 검출. COCO pretrained
- **Motion Blob(모션 블롭):** 프레임 차분으로 움직이는 영역 bbox
- **Hybrid Detection(하이브리드 디텍션):** IPAD=motion 주 · 합성=YOLO+motion
- **NMS(엔엠에스):** Non-Maximum Suppression — 겹치는 박스 제거
- **COCO(코코):** 80클래스 범용 객체 데이터셋

### Detection 필드

| 필드 | 의미 |
|------|------|
| `bbox` | (x1,y1,x2,y2) 픽셀 |
| `confidence` | 0~1 |
| `source` | `"yolo"` 또는 `"motion"` |
| `class_name` | IPAD 부품 → `"motion_part"` |

### 소스코드 핵심

| 함수 | 역할 |
|------|------|
| `get_yolo_model()` | lazy-load YOLOv8n |
| `detect_objects_yolo()` | COCO 클래스 필터 |
| `detect_motion_blobs()` | operation_zone 내 프레임 차분 |
| `detect_objects_hybrid()` | IPAD/합성 분기 |
| `_operation_roi_mask()` | ROI 마스크 |

> 산업 부품은 COCO에 없음 → **형태 몰라도 움직임**으로 검출.

---

## Part 3 — ByteTrack 다중 객체 추적 (MOT)

### 이 섹션에서 배우는 것

- 프레임이 바뀌어도 **같은 track_id**를 유지한다.
- "ID3번 부품이 구역을 벗어났다"처럼 **시간에 따른 행동**을 설명한다.

### 핵심 용어

- **MOT(엠오티, Multi-Object Tracking):** 다중 객체 추적 — 프레임 간 **ID 유지**
- **ByteTrack(바이트트랙):** 저신뢰 검출도 2차 매칭 → **occlusion(가림)** 에 강함
- **IoU(아이오유):** bbox 겹침 0~1 — 매칭 점수
- **Track(트랙):** track_id + bbox + confidence + frame_index
- **SimpleByteTracker:** 교육용 IoU 기반 ByteTrack 아이디어 구현

### ByteTrack 2단계 매칭

1. **고신뢰** 검출 → 기존 track IoU 매칭  
2. **저신뢰** 검출 → 2차 매칭 (잠깐 가려져도 ID 유지)

### 경로 분기

| 데이터 | 추적 방식 |
|--------|-----------|
| IPAD | motion hybrid + `SimpleByteTracker` |
| 합성/기타 | Ultralytics `model.track(bytetrack.yaml)` |

### 소스코드 핵심

| 함수/클래스 | 역할 |
|-------------|------|
| `bbox_iou()` | IoU 계산 |
| `SimpleByteTracker.update()` | 고/저신뢰 2단계 매칭 |
| `track_objects_bytetrack()` | IPAD motion hybrid 강제 |
| `track_objects_bytetrack_ultra()` | YOLO 내장 ByteTrack |
| `track_series[i]` | i번째 프레임 Track 리스트 |

---

## Part 4 — 설비 동작 상태 분류

### 이 섹션에서 배우는 것

- 추적·motion 신호로 프레임마다 **RUNNING / IDLE / ANOMALY** 라벨을 붙인다.

### 핵심 용어

- **ROI Zone(알오아이 존):** Region of Interest — 설비 동작 허용 구역
- **Critical Zone(크리티컬 존):** 부품이 **있어야 정상**인 operation_zone
- **Track Velocity(트랙 벨로시티):** bbox 중심 이동 px/frame
- **Operation State(오퍼레이션 스테이트):** 가동/대기/이상 상태 라벨
- **Zone Violation(존 바이올레이션):** operation_zone **이탈**
- **Motion Z-score(모션 지-스코어):** baseline 대비 동작 급변 (02·07 동일)

### 상태 분류

| 상태 | 의미 | 대표 조건 |
|------|------|-----------|
| `RUNNING` | 가동 중 | velocity ≥ 3.5 px/frame |
| `IDLE` | 대기·저활동 | velocity ≤ 0.8 px/frame |
| `ANOMALY` | 이상 후보 | 구역 연속 이탈 또는 motion_z ≥ 2.2 |

### EQUIPMENT_ZONES (비율 0~1)

- `operation_zone`: (0.20, 0.25, 0.80, 0.85) — 정상 작업 영역
- `idle_zone`: (0.02, 0.55, 0.18, 0.95) — 대기 참고

### 소스코드 핵심

| 함수 | 역할 |
|------|------|
| `compute_motion_energy()` | 인접 프레임 그레이 차이 |
| `motion_zscore()` | baseline z-score |
| `compute_track_velocity()` | track_id별 중심 이동 |
| `check_zone_violations()` | operation_zone 밖 motion_part/person |
| `classify_equipment_state()` | RUNNING/IDLE/ANOMALY |
| `FrameState` | 프레임별 관제 레코드 dataclass |

---

## Part 5 — 상황 요약 에이전트 (Global Reasoning)

### 이 섹션에서 배우는 것

- 교재 **Video-LLaVA** 단계의 **경량 대체**: 이벤트 → **한국어 관제 문장**.

### 핵심 용어

- **Situation Report(시츄에이션 리포트):** 관제원이 읽을 **한국어 상황 문장**
- **Situation Agent(시츄에이션 에이전트):** FrameState → 자연어 변환
- **Template Agent(템플릿 에이전트):** 규칙·패턴 기반 문장 (VLM 대체)
- **Global Reasoning(글로벌 리저닝):** 영상 맥락 전체 이해 (교재 VLM 단계)

### 리포트 생성 규칙

- `ANOMALY` 프레임: **즉시** 리포트 (교재 "VLM 에이전트 활성화")
- 그 외: 클립 길이의 약 **1/6 간격** 샘플링

### 소스코드

- `generate_situation_report()` — 상태별 한국어 문장
- `SituationReport` — report_id, frame_index, summary_ko, details

> 실무: 05번 VLM + Video-LLaVA API로 품질 고도화.

---

## Part 6 — ChromaDB · 시맨틱 관제 검색

### 이 섹션에서 배우는 것

- 상황 요약을 **벡터 DB**에 저장하고 자연어로 과거 사건을 검색한다.

### 핵심 용어

- **Vector DB(벡터 디비):** 문장을 숫자 벡터(embedding)로 저장
- **Semantic Search(시맨틱 서치):** 키워드 불일치해도 **의미 유사** 검색
- **Cosine Similarity(코사인 유사도):** 벡터 방향 유사도
- **Embedding(임베딩):** 텍스트 → 고차원 벡터

예: "컨베이어 멈춤" → "저활동/대기 상태" 리포트 매칭

### 구성

| 구성요소 | 설명 |
|----------|------|
| `_hash_embed()` | 교육용 결정적 벡터 (실무: sentence-transformers) |
| `InMemoryReportIndex` | ChromaDB 없을 때 코사인 검색 |
| ChromaDB | `USE_CHROMA=1` 시 `output_tracking/chroma_db/` 영속 저장 |

### 소스코드

- `index_reports_chroma()`, `search_reports()`
- 데모 질의: "설비 구역 이탈", "정상 가동", "동작 급변 이상"

---

## Part 7 — 관제 대시보드 · MOT 지표 · manifest

### 이 섹션에서 배우는 것

- 한 세션 결과를 **차트**로 요약하고 `tracking_manifest.json`을 export한다.

### MOT 성능 지표

| 지표 | 의미 |
|------|------|
| **ID Switches** | 주요 track_id 변경 횟수 — 낮을수록 안정 |
| **Track Continuity** | track_id 평균 유지 프레임 수 |
| **Processing FPS** | 초당 처리 프레임 — 실시간 관제 여유 |

### 대시보드 구성

- 상태 파이·막대 (RUNNING/IDLE/ANOMALY)
- Motion Energy 타임라인
- 프레임별 활성 Track 수
- MOT 지표 텍스트

### 소스코드 핵심

| 함수 | 역할 |
|------|------|
| `count_id_switches()` | ID 변경 횟수 |
| `track_continuity()` | 평균 관측 프레임 |
| `evaluate_tracking_session()` | MOT·FPS dict |
| `plot_control_dashboard()` | 통합 PNG |
| `export_tracking_manifest()` | JSON v1 |

---

## 부록: 출력 파일

| 파일 | 내용 |
|------|------|
| `figures/01_source_overview.png` | 원본·ROI·motion |
| `predictions/` | 검출·추적 오버레이 |
| `figures/07_control_dashboard.png` | 관제 대시보드 |
| `chroma_db/` | ChromaDB (USE_CHROMA=1) |
| `tracking_manifest.json` | 프레임별 추적·상태·요약 |

---

## 08번 통합 manifest 연동

1. `MANIFEST_SOURCES`에 `output_tracking/tracking_manifest.json` 추가
2. `operation_state`, `motion_z` 신호를 unified_score에 병합
3. 05번 VLM으로 `ANOMALY` 키프레임 자연어 요약 고도화
4. 실 CCTV RTSP → decord 스트림으로 PoC 확장

---

## FAQ

**Q. YOLO가 아무것도 안 잡혀요 (IPAD).**  
A. 정상. IPAD는 motion blob이 **주 검출**입니다.

**Q. ultralytics 설치 실패?**  
A. `SimpleByteTracker` + motion hybrid로 Part 3~7 계속 가능.

**Q. ChromaDB crash (Windows)?**  
A. 기본은 `InMemoryReportIndex`. `USE_CHROMA=1`은 선택.

---

*본 문서는 `09_Video_Object_Tracking_Equipment_State.ipynb` 강의용으로 작성되었습니다.*
