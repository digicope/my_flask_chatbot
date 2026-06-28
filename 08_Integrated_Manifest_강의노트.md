# 08 Integrated Manifest — 강의용 노트

> **대상:** Video AI를 처음 접하는 초보자 (03~07 노트북 수강 후)  
> **원본:** `08_Integrated_Manifest.ipynb`  
> **실습 데이터:** `IPAD_sample.zip` + 03~07 산출 manifest JSON

---

## 강의 전체 흐름

| Part | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1 | manifest 소스 탐색 | 03~07 JSON 존재 확인 · IPAD 타임라인 |
| 2 | 신호 추출(normalize) | 소스별 필드 파싱 → `FrameRecord` |
| 3 | 프레임 조인 · 통합 스코어 | `merge_frame_records` · 알람 후보 |
| 4 | export · 대시보드 | `unified_manifest.json` · CSV · 차트 |
| 5 | Manifest Query API | 조건 검색 헬퍼 |
| 6 | 상관 · 공발생 분석 | 파이프라인 합의도 · 신호 상관 |
| 7 | 알람 썸네일 모자이크 | 육안 검증용 그리드 |

---

## 통합 Manifest란?

**한 줄 요약:** 여러 AI 모듈(샘플링, 이상탐지, OCR, 키포인트 등)이 각자 만든 JSON을 **프레임 단위로 한데 모아** `unified_manifest.json` 하나로 정리합니다.

```
03 샘플링  →  "중요 프레임 8, 38, 179번"
04 이상탐지 →  "0~12번 구간 수상"
06 OCR     →  "42번에 'WARNING' 텍스트"
07 키포인트 →  "128번 motion_z 급증"
        ↓ frame_key 조인
unified_manifest.json + CSV + 차트
```

### 통합 대상 manifest

| 노트북 | 파일 | 주요 신호 |
|--------|------|-----------|
| 03 | `output_sampling/sampling_manifest.json` | swin/detr 선별 프레임 |
| 05 | `output_vlm_summary/summary_manifest.json` | vlm_frames, text_summary |
| 04 | `output_anomaly_scene/anomaly_manifest.json` | anomaly_scenes, top_frame_indices |
| 06 | `output_ocr_kr/ocr_manifest_ipad.json` | full_text, detections |
| 07 | `output_keypoint/keypoint_manifest.json` | displacement, motion_z |

> 선행 실행: 03~07 중 일부만 실행해도 **있는 manifest만** 병합. `[MISS]`는 정상.

---

## Part 1 — manifest 소스 탐색 · IPAD 타임라인

### 이 섹션에서 배우는 것

- 통합 전 **재료 준비 상태**를 스캔한다.
- 모든 파이프라인이 공유할 **IPAD testing 프레임 목록**을 불러온다.

### 핵심 용어

- **Manifest(매니페스트):** AI 분석 결과 JSON. 영상 전체가 아니라 **메타데이터·인덱스·점수** 위주
- **Manifest Registry(매니페스트 레지스트리):** 병합할 JSON 경로 목록 — `MANIFEST_SOURCES`
- **Coverage(커버리지):** 어떤 소스가 있고 없는지 기록한 **준비 상태 리포트**
- **Schema Validator(스키마 밸리데이터):** 소스별 JSON **필수 키** 검사 — `validate_manifest_schema`
- **Frame Key(프레임 키):** 프레임 조인용 **고유 식별자** — `normalize_frame_key`
- **Primary Key(프라이머리 키):** DB에서 행을 합치는 기본 키와 같은 개념

### frame_key가 왜 중요한가?

```
C:\...\IPAD_Sample\R01\testing\frames\01\042.jpg
                ↓ normalize_frame_key
         frames/01/042.jpg   ← 모든 소스가 이 키로 조인
```

- sampling·anomaly는 `frame_index=42` (정수)
- OCR·keypoint는 `C:\...\042.jpg` (경로)
- IPAD 프레임 목록으로 **번호 ↔ 경로** 연결

### 소스코드 핵심

| 함수/변수 | 역할 |
|-----------|------|
| `MANIFEST_SOURCES` | sampling, vlm_summary, anomaly_scene, ocr_*, keypoint 경로 |
| `load_json_manifest()` | JSON 읽기. 없거나 파싱 실패 시 `None` |
| `load_ipad_testing_frames()` | testing 클립 프레임 경로 정렬 반환 |
| `normalize_frame_key()` | 경로 표기를 `frames/클립/파일명` 형태로 통일 |
| `frame_key_from_index()` | 정수 인덱스 → frame_key |
| `LoadedManifest` | 로드된 manifest 한 건 (name, path, data) |

**출력 읽는 법**
- `[ OK ]` — 병합 포함
- `[MISS]` — 건너뜀
- `[Schema] 경고` — 구조 이상, 결과 신뢰도 주의

**생성:** `output_unified/manifest_coverage.json`

---

## Part 2 — 소스별 신호 추출(normalize)

### 이 섹션에서 배우는 것

- 제각각인 JSON 형식을 **공통 `FrameRecord`**로 변환한다.
- 프레임별 **신호(signal)**만 추출해 `frame_store`에 누적한다.

### 핵심 용어

- **Signal(신호):** 프레임 하나에 붙는 분석 결과 (예: `anomaly_candidate=True`, `motion_z=2.3`)
- **Normalize(노멀라이즈, 정규화):** 서로 다른 형식을 **공통 스키마**로 맞춤
- **Signal Extractor(시그널 익스트랙터):** manifest 종류별 파싱 함수
- **FrameRecord(프레임 레코드):** 프레임 하나의 통합 레코드 (신호·소스 목록)
- **_clip_meta:** 프레임이 아닌 **클립 전체** 정보 (VLM 요약문, anomaly 장면 목록)

### 소스별 추출 내용

| 소스 | 프레임 신호 | 클립 메타 |
|------|------------|-----------|
| sampling | `sampling_swin`, `sampling_detr` | — |
| vlm_summary | `vlm_keyframe` | text_summary, segments |
| anomaly_scene | `anomaly_candidate`, `anomaly_top` | anomaly_scenes, text_report |
| ocr_* | `ocr_full_text`, `ocr_detection_count` | — |
| keypoint | `displacement`, `motion_z` | motion_spike_frames |

### 소스코드 핵심

| 함수 | 역할 |
|------|------|
| `_ensure_record()` | frame_key별 FrameRecord 생성·반환 |
| `extract_sampling_signals()` | preset별 선별 인덱스 → 불리언 신호 |
| `extract_vlm_signals()` | vlm_frames + 클립 요약 |
| `extract_anomaly_signals()` | top/anomaly 인덱스 → 플래그 |
| `extract_ocr_signals()` | image_path 기준 조인 |
| `extract_keypoint_signals()` | displacement·motion_z 시계열 |
| `EXTRACTORS` | manifest 이름 → extractor 매핑 |

> **강의 포인트:** IPAD 타임라인 전체에 `frame_index`, `time_sec`를 채워 두면 신호 없는 구간도 **연속 시계열**로 표현 가능.

---

## Part 3 — 프레임 조인 · unified_score · 알람 후보

### 이 섹션에서 배우는 것

- 여러 신호를 **하나의 점수**로 합산한다.
- 알람 후보 프레임을 선정한다.

### 핵심 용어

- **Unified Score(유니파이드 스코어):** anomaly, VLM, sampling, motion, OCR **가중 합산** 0~1 점수
- **Signal Agreement(시그널 어그리먼트):** 5개 파이프라인 중 몇 개가 "주목 프레임"에 동의하는지 (0~1)
- **Join Engine(조인 엔진):** `merge_frame_records` — OCR 중복 해소 등
- **Alert Candidate(알람 후보):** score ≥ threshold 또는 상위 K개 프레임

### SCORE_WEIGHTS (기본값)

| 신호 | 가중치 |
|------|--------|
| anomaly_candidate | 0.45 |
| anomaly_top | 0.10 |
| vlm_keyframe | 0.20 |
| sampling | 0.10 |
| motion_z_high (≥2.0) | 0.20 |
| ocr_detected | 0.05 |
| agreement 보너스 | +0.05 × agreement |

- `ALERT_SCORE_THRESHOLD = 0.55`
- `ALERT_TOP_K = 8`
- `MOTION_Z_THRESHOLD = 2.0`

### OCR 우선순위

같은 프레임에 `ocr_ipad`와 `ocr_hub`가 둘 다 있으면 **`ocr_ipad` 우선** (실제 설비 영상에 가깝기 때문).

### 소스코드 핵심

| 함수 | 역할 |
|------|------|
| `compute_signal_agreement()` | 5개 독립 투표 → 0~1 |
| `merge_frame_records()` | OCR 우선순위, source dedupe |
| `compute_unified_score()` | 가중 합산, max 1.0 cap |
| `build_unified_records()` | FrameRecord → JSON list |
| `select_alert_candidates()` | threshold + top-K |

---

## Part 4 — export · 커버리지 대시보드

### 핵심 용어

- **Master JSON(마스터 제이슨):** `unified_manifest.json` — 프로그램·API 정본
- **Flat CSV(플랫 씨에스브이):** 한 행 = 한 프레임. Excel·Power BI용
- **utf-8-sig:** Excel 한글 깨짐 방지 BOM

### 출력물

| 파일 | 용도 |
|------|------|
| `unified_manifest.json` | 앱 연동 정본 |
| `unified_frames.csv` | 스프레드시트 분석 |
| `01_coverage_heatmap.png` | 프레임×신호 히트맵 |
| `02_alert_candidates.png` | 알람 후보 요약표 |
| `03_unified_timeline.png` | score 시계열 + 이벤트 마커 |
| `04_source_overlap.png` | 소스 커버리지·공발생 |

### 소스코드 핵심

| 함수 | 역할 |
|------|------|
| `export_unified_manifest()` | JSON 저장 |
| `export_frames_csv()` | 플랫 CSV |
| `plot_unified_timeline()` | score + anomaly/vlm/motion 마커 |
| `plot_source_overlap()` | 막대 + co-occurrence 히트맵 |
| `plot_manifest_coverage()` | 프레임×신호 히트맵 |

**타임라인 읽는 법:** 파란 선 = unified_score · 빨간 점선 = threshold · scatter = anomaly(x), vlm(o), motion(^)

---

## Part 5 — Manifest Query API

### 이 섹션에서 배우는 것

- JSON 전체를 매번 파싱하지 않고 **조건 검색**하는 실무용 인터페이스.

### 핵심 용어

- **Manifest Query(매니페스트 쿼리):** unified manifest **읽기 인터페이스** (SQL WHERE와 유사)

### ManifestQuery 메서드

| 메서드 | 역할 |
|--------|------|
| `by_score(min)` | 점수 이상 필터 |
| `by_source(name)` | 특정 파이프라인 기여 프레임 |
| `by_time_range(s, e)` | 초 단위 구간 |
| `with_ocr_keyword(kw)` | OCR 텍스트 키워드 |
| `top_alerts(k)` | 상위 k개 |
| `summary()` | 평균·최대 score, 소스 목록 |

---

## Part 6 — 다중 신호 상관 · 공발생

### 핵심 용어

- **Binary Signal(이진 신호):** 프레임별 0/1 단순화
- **Pearson Correlation(피어슨 상관):** 두 신호 동조 정도 (-1~1)
- **Co-occurrence(코어커런스, 공발생):** 같은 프레임에 두 소스 **동시** 기록 횟수

### 해석 가이드

- `anomaly ~ motion_z` 상관 ↑ → 이상 탐지와 움직임 급증 **같은 구간**
- `vlm_keyframe` 상관 ↓ → VLM 키프레임이 다른 모듈 관심과 **다를 수 있음**
- `high_agreement_frames` ↑ → 다중 파이프라인 **신뢰도 높은 구간** 많음

**생성:** `signal_cooccurrence.json`, `05_agreement_histogram.png`

---

## Part 7 — 알람 후보 썸네일 모자이크

### 핵심 용어

- **Thumbnail(썸네일):** 원본보다 작은 미리보기
- **Mosaic(모자이크):** 썸네일 격자 합성
- **imdecode:** Windows 한글 경로 안전 읽기

> 숫자·표만으로는 알람 후보 화면을 직관적으로 파악하기 어렵습니다. **1차 육안 점검**용.

**생성:** `figures/06_alert_mosaic.jpg` — 타일 상단 `#프레임번호 s=점수`

### 소스코드

- `read_frame_bgr()`, `build_alert_mosaic()`

---

## FAQ

**Q. manifest가 대부분 MISSING?**  
A. 해당 번호 노트북(03~07)을 먼저 실행하세요.

**Q. unified_score 조정?**  
A. `SCORE_WEIGHTS`, `ALERT_SCORE_THRESHOLD` 수정. `signal_agreement` 높은 프레임은 다중 합의 구간.

**Q. OCR ipad·hub 둘 다?**  
A. `ocr_ipad` 우선.

---

## 시리즈 연계

- **입력:** 03~07 manifest
- **확장:** 09 `tracking_manifest.json`을 `MANIFEST_SOURCES`에 추가 가능

---

*본 문서는 `08_Integrated_Manifest.ipynb` 강의용으로 작성되었습니다.*
