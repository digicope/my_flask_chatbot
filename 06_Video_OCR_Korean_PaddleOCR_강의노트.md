# 06 Video OCR Korean PaddleOCR — 강의용 노트

> **대상:** Video AI 초보자  
> **원본:** `06_Video_OCR_Korean_PaddleOCR.ipynb`  
> **실습 데이터:** `AI_Hub_한글_OCR.zip` → `AI_Hub_한글_OCR/`

---

## 강의 전체 흐름

| Part | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1 | AI Hub 데이터 | JSON 라벨 · GT 시각화 |
| 2 | Detection | Morphology proposal · IoU 평가 |
| 3 | PaddleOCR | pretrained det+rec · CER |
| 4 | 통합 파이프라인 | manifest · IPAD 프레임 OCR |

---

## OCR이란?

```
이미지/프레임
  ↓ Detection — "글자가 어디?" (bbox)
  ↓ Recognition — "무슨 글자?" (문자열)
  ↓ bbox + text + confidence
```

영상 OCR = **프레임마다 2D OCR 반복**. VLM 요약과 달리 **좌표·정확한 문자열** 필요.

---

## Part 1 — AI Hub 한글 OCR 데이터셋

### 핵심 용어

- **OCR(오씨알, Optical Character Recognition):** 광학 문자 인식
- **Scene Text(씬 텍스트):** 간판·HMI·현장 촬영 텍스트 (책 스캔 아님)
- **GT(지티, Ground Truth):** 사람이标注한 정답 bbox + text
- **TextBox(텍스트박스):** 4점 quad + transcription
- **wordbox(워드박스):** AI Hub JSON `[x,y,w,h]` + `value`
- **원천데이터 / 라벨링데이터:** jpg와 json **별도 폴더**
- **Don't care(돈트 케어):** `xxx` — 평가 제외 영역

### 데이터 구조

```
AI_Hub_한글_OCR/
  01.원천데이터/.../*.jpg
  02.라벨링데이터/.../*.json
```

### 소스코드

- `ensure_aihub_ocr_dataset()`, `load_aihub_dataset()`
- `parse_aihub_json()` → `OcrSample` 리스트
- `plot_ocr_samples()` — 초록 GT polygon + 노란 라벨
- 저장: `01_train_gt_overlay.png`

> **인코딩:** zip cp949 · JSON utf-8/cp949 · `np.fromfile`+`imdecode`

---

## Part 2 — 텍스트 검출 · Morphology Proposal

### 핵심 용어

- **Detection(디텍션, 검출):** 글자 **위치(bbox)** 만 찾기
- **Recognition(레코그니션, 인식):** crop 영역에서 **문자열** 읽기
- **Proposal(프로포절):** 알고리즘이 추측한 bbox 후보
- **IoU(아이오유, Intersection over Union):** 두 bbox 겹침 0~1. ≥0.5면 "맞춤"
- **Precision(프리시전):** proposal 중 GT 맞은 비율 — **허위 alarm** 적을수록 ↑
- **Recall(리콜):** GT 중 proposal이 잡은 비율 — **놓친 글자** 적을수록 ↑
- **F1(에프원):** P·R 조화 평균
- **Morphology(몰폴로지):** adaptive threshold + close → contour bbox (전통 방식)

### OpenCV Proposal 파이프라인

```
그레이 → GaussianBlur → adaptiveThreshold → MORPH_CLOSE → findContours → bbox
```

> **교육용 베이스라인.** 복잡한 한글 문서에서는 F1 낮음 → Part 3 PaddleOCR 비교

### 소스코드

- `propose_text_regions()`, `polygon_iou()`, `evaluate_detection()`
- 저장: `02_detection_proposal.png`

---

## Part 3 — PaddleOCR (pretrained)

### 핵심 용어

- **PaddleOCR(패들오씨알):** Baidu 오픈소스 OCR. **한글 det+rec** pretrained
- **pretrained(프리트레인드):** 직접 학습 없이 **사전학습 weight** 사용
- **DB(디비):** Differentiable Binarization — PaddleOCR 검출 백엔드
- **Angle Classifier(앵글 클래시파이어):** 기울어진 글자 방향 보정
- **CRNN(씨알알앤)/SVTR:** 인식 백엔드
- **CER(씨이알, Character Error Rate):** 글자 단위 오류율. 0=완벽, 1=전부 틀림
- **Levenshtein(레벤슈타인):** 편집 거리 — CER 계산 기반
- **confidence(컨피던스):** 모델 신뢰도 0~1

### 설치

```bash
pip install paddlepaddle==2.6.2 paddleocr==2.9.1
# GPU: paddlepaddle-gpu (CUDA 버전 맞춤)
```

### 소스코드

- `get_paddle_ocr()` — 싱글톤 lazy load
- `run_paddle_ocr()` — det → (angle cls) → rec
- `evaluate_paddle_on_gt()` — IoU 매칭 + CER
- `PADDLE_LANG = "korean"`

> **첫 실행:** 한글 모델 weight 자동 다운로드 (수백 MB)

---

## Part 4 — 통합 OCR 파이프라인

### 파이프라인

```
이미지 / IPAD 프레임 → PaddleOCR → ocr_manifest.json + predictions/*.png
```

### manifest JSON 구조

```json
{
  "engine": "PaddleOCR",
  "results": [{
    "image_path": "...",
    "full_text": "글자1 | 글자2",
    "detections": [{"bbox": [...], "text_pred": "...", "confidence": 0.95}]
  }]
}
```

### 소스코드

- `run_ocr_pipeline()`, `export_ocr_manifest()`
- `_find_ipad_frames()` — 05·03 IPAD 연계
- `ocr_manifest.json`, `ocr_manifest_ipad.json`

---

## 05 VLM vs 06 OCR

| | VLM | OCR |
|---|-----|-----|
| 출력 | 자연어 요약 | **정확한 문자열+좌표** |
| 용도 | 장면 설명 | HMI·라벨·표 읽기 |

---

## FAQ

**Q. Part 2 F1=0?**  
A. Morphology 한계. Part 3 PaddleOCR 결과를 보세요.

**Q. OCR이 느려요.**  
A. CPU는 이미지당 수십 초. GPU paddlepaddle 권장.

**Q. zip 폴더명 깨짐?**  
A. Windows CP949 — `_extract_aihub_zip` cp949 복원.

---

## 부록: 출력 파일

| 파일 | 내용 |
|------|------|
| `output_ocr_kr/figures/01_*_gt_overlay.png` | GT 시각화 |
| `output_ocr_kr/figures/02_detection_proposal.png` | GT vs proposal |
| `output_ocr_kr/figures/03_paddleocr_*.png` | PaddleOCR 결과 |
| `output_ocr_kr/ocr_manifest.json` | AI Hub OCR JSON |
| `output_ocr_kr/ocr_manifest_ipad.json` | IPAD OCR JSON |

---

*본 문서는 `06_Video_OCR_Korean_PaddleOCR.ipynb` 강의용으로 작성되었습니다.*
