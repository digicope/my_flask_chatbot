# 웹 챗봇 프로젝트

Flask와 OpenAI API를 사용한 카카오톡 스타일의 웹 챗봇 애플리케이션입니다.

## 📋 목차

- [프로젝트 소개](#프로젝트-소개)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설치 방법](#설치-방법)
- [환경 변수 설정](#환경-변수-설정)
- [실행 방법](#실행-방법)
- [사용 방법](#사용-방법)

## 프로젝트 소개

이 프로젝트는 OpenAI의 GPT-4o-mini 모델을 활용한 웹 기반 챗봇 서버입니다. 카카오톡과 유사한 직관적인 UI를 제공하며, 실시간으로 AI와 대화할 수 있습니다.

## 주요 기능

- ✅ **카카오톡 스타일 UI**: 카카오톡 채팅방과 유사한 디자인
- ✅ **실시간 채팅**: 새로고침 없이 비동기 통신으로 대화 가능
- ✅ **OpenAI GPT-4o-mini**: 최신 OpenAI API를 사용한 자연스러운 대화
- ✅ **반응형 디자인**: 화면 중앙 정렬, 모바일 친화적 레이아웃
- ✅ **메시지 타임스탬프**: 각 메시지에 시간 표시
- ✅ **고정 입력창**: 하단 입력창 고정, 메시지 영역만 스크롤

## 기술 스택

### Backend
- **Flask**: Python 웹 프레임워크
- **OpenAI API**: GPT-4o-mini 모델 사용
- **python-dotenv**: 환경 변수 관리

### Frontend
- **HTML5**: 마크업
- **CSS3**: 스타일링 및 애니메이션
- **JavaScript (Vanilla)**: Fetch API를 사용한 비동기 통신

## 프로젝트 구조

```
Web_Chatbot_Project/
│
├── app.py                 # Flask 서버 메인 파일
├── requirements.txt       # Python 패키지 의존성
├── .env                   # 환경 변수 파일 (생성 필요)
├── README.md             # 프로젝트 문서
│
├── templates/
│   └── index.html        # 메인 HTML 템플릿
│
└── static/
    └── style.css         # CSS 스타일시트
```

## 설치 방법

### 1. 저장소 클론 또는 다운로드

프로젝트 파일을 로컬에 다운로드합니다.

### 2. Python 가상 환경 생성 (권장)

```bash
python -m venv venv
```

### 3. 가상 환경 활성화

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. 패키지 설치

```bash
pip install -r requirements.txt
```

## 환경 변수 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 OpenAI API 키를 설정합니다.

```env
OPENAI_API_KEY=your_openai_api_key_here
```

> **참고**: OpenAI API 키는 [OpenAI Platform](https://platform.openai.com/api-keys)에서 발급받을 수 있습니다.

## 실행 방법

### 1. 서버 실행

```bash
python app.py
```

서버가 성공적으로 실행되면 다음과 같은 메시지가 표시됩니다:

```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### 2. 브라우저에서 접속

웹 브라우저를 열고 다음 주소로 접속합니다:

```
http://localhost:5000
```

또는

```
http://127.0.0.1:5000
```

## 사용 방법

1. **메시지 입력**: 하단 입력창에 메시지를 입력합니다.
2. **전송**: 전송 버튼을 클릭하거나 `Enter` 키를 누릅니다.
3. **대화**: 챗봇의 응답이 자동으로 표시됩니다.
4. **스크롤**: 메시지가 많아지면 채팅 영역만 스크롤됩니다.

## 주요 특징

### UI/UX
- 카카오톡과 유사한 디자인
- 사용자 메시지: 오른쪽 정렬, 노란색 말풍선
- 챗봇 메시지: 왼쪽 정렬, 흰색 말풍선
- 각 메시지에 시간 표시 (오전/오후 형식)
- 부드러운 애니메이션 효과

### 기술적 특징
- RESTful API 설계
- 비동기 Fetch API 사용
- 에러 처리 및 사용자 피드백
- 환경 변수를 통한 안전한 API 키 관리

## 문제 해결

### API 키 오류
- `.env` 파일이 프로젝트 루트에 있는지 확인
- `OPENAI_API_KEY` 변수명이 정확한지 확인
- API 키가 유효한지 확인

### 포트 충돌
- 기본 포트 5000이 사용 중인 경우, `app.py`의 마지막 줄을 수정:
  ```python
  app.run(debug=True, port=5001)  # 다른 포트 번호 사용
  ```

### 패키지 설치 오류
- Python 버전 확인 (Python 3.7 이상 권장)
- pip 업그레이드: `python -m pip install --upgrade pip`

## 라이선스

이 프로젝트는 개인 학습 및 개발 목적으로 자유롭게 사용할 수 있습니다.

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

---

**개발자**: AI Cursor Lab  
**버전**: 1.0.0  
**최종 업데이트**: 2026년 1월
