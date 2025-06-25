# 로컬 LLM 기반 회사 가이드 챗봇 API

로컬에서 직접 모델을 로드하여 동작하는 RAG(Retrieval-Augmented Generation) 기반 회사 가이드 챗봇 API입니다.

## 🚀 주요 특징

- 🤖 **로컬 LLM**: 외부 API 없이 로컬에서 직접 모델 실행
- 📚 **RAG 시스템**: 문서 기반 정확한 답변 생성
- 🔍 **벡터 검색**: FAISS를 활용한 고성능 유사도 검색
- 🌐 **웹 인터페이스**: 직관적인 웹 UI 제공
- 📄 **다양한 문서 형식**: TXT, MD, PDF, DOCX 지원
- 🎯 **신뢰도 점수**: 응답의 신뢰도를 수치로 제공

## 🛠️ 기술 스택

- **FastAPI**: 현대적이고 빠른 웹 프레임워크
- **LangChain**: LLM 애플리케이션 개발 프레임워크
- **Transformers**: Hugging Face 로컬 모델 로드
- **FAISS**: 효율적인 벡터 검색 라이브러리
- **Sentence Transformers**: 다국어 텍스트 임베딩
- **PyTorch**: 딥러닝 프레임워크

## 📋 시스템 요구사항

- Python 3.11 이상
- 최소 4GB RAM (모델 로드용)
- 인터넷 연결 (초기 모델 다운로드용)

## 🚀 빠른 시작

### 1. 프로젝트 설정

```bash
# 저장소 클론
git clone <repository-url>
cd simple-chatbot

# 자동 설정 스크립트 실행
./setup.sh
```

### 2. 가상환경 활성화

```bash
source venv/bin/activate
```

### 3. FastAPI CLI로 서버 실행

#### 방법 1: 자동 스크립트 사용 (권장)

```bash
./start.sh
```

#### 방법 2: 직접 uvicorn 명령어 사용

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

#### 방법 3: 개발 모드 (더 자세한 로그)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

### 4. 웹 브라우저에서 접속

- **웹 인터페이스**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs
- **ReDoc 문서**: http://localhost:8000/redoc

## 📖 사용 방법

### 1. 문서 업로드

웹 인터페이스에서 파일을 드래그 앤 드롭하거나 파일 선택 버튼을 클릭하여 문서를 업로드합니다.

### 2. 질문하기

업로드된 문서에 대해 자연어로 질문을 입력하면 RAG 시스템이 관련 정보를 찾아 답변을 생성합니다.

### 3. API 사용

```python
import requests

# 채팅 요청
response = requests.post("http://localhost:8000/chat", json={
    "message": "회사는 언제 설립되었나요?",
    "user_id": "user123"
})

result = response.json()
print(f"응답: {result['response']}")
print(f"소스: {result['sources']}")
print(f"신뢰도: {result['confidence']}")
```

## 🔧 API 엔드포인트

### 채팅

- **POST** `/chat` - 질문에 대한 RAG 기반 응답 생성

### 문서 관리

- **POST** `/upload-document` - 문서 업로드 및 벡터화
- **GET** `/documents` - 저장된 문서 목록 조회
- **DELETE** `/documents/{document_id}` - 문서 삭제

### 시스템

- **GET** `/health` - 시스템 상태 확인
- **GET** `/` - 웹 인터페이스

## 🧪 테스트

```bash
# API 테스트 실행
python test_api.py
```

## 📁 프로젝트 구조

```
simple-chatbot/
├── app.py                 # FastAPI 메인 애플리케이션
├── rag_system.py          # RAG 시스템 핵심 로직
├── requirements.txt       # Python 의존성
├── pyproject.toml         # 프로젝트 설정 (FastAPI CLI용)
├── setup.sh              # 자동 설정 스크립트
├── start.sh              # FastAPI CLI 실행 스크립트
├── test_api.py           # API 테스트 스크립트
├── static/               # 웹 인터페이스
│   └── index.html
├── example_documents/    # 샘플 문서
│   └── company_guide.txt
└── README.md
```

## 🔍 RAG 시스템 작동 원리

1. **문서 처리**: 업로드된 문서를 의미 있는 청크로 분할
2. **벡터화**: 각 청크를 다국어 임베딩 모델로 벡터화
3. **검색**: 사용자 질문과 유사한 문서 청크 검색
4. **생성**: 로컬 LLM이 검색된 컨텍스트를 바탕으로 응답 생성
5. **소스 추적**: 응답의 출처 문서 정보 제공

## ⚙️ 모델 설정

### 기본 모델

- **주 모델**: `microsoft/DialoGPT-medium` (한국어 지원)
- **대체 모델**: `distilgpt2` (더 가벼운 모델)

### 모델 변경

`rag_system.py` 파일에서 `model_name` 변수를 수정하여 다른 모델을 사용할 수 있습니다.

## 🐛 문제 해결

### 모델 로드 실패

- 인터넷 연결 확인
- 충분한 메모리 확보 (최소 4GB)
- PyTorch 버전 호환성 확인

### 성능 최적화

- 더 가벼운 모델 사용 (`distilgpt2`)
- GPU 사용 (CUDA 지원 시)
- 청크 크기 조정

### FastAPI CLI 관련

- `uvicorn` 패키지가 설치되어 있는지 확인
- 포트 8000이 사용 중인 경우 다른 포트 사용: `--port 8001`

## 📝 라이센스

MIT License

## 🤝 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

## 📞 지원

문제가 발생하면 이슈를 생성해주세요.
