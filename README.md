# RAG 챗봇 API

LangChain을 사용한 RAG(Retrieval-Augmented Generation) 기반 챗봇 API입니다.

## 주요 기능

- 📚 **문서 업로드 및 벡터화**: 텍스트 문서를 업로드하여 벡터 데이터베이스에 저장
- 🔍 **유사도 검색**: 사용자 질문과 관련된 문서 청크를 검색
- 🤖 **지능형 응답**: 검색된 문서를 바탕으로 정확한 답변 생성
- 📊 **소스 추적**: 응답의 출처 문서 정보 제공
- 🎯 **신뢰도 점수**: 응답의 신뢰도를 수치로 제공

## 기술 스택

- **FastAPI**: 현대적이고 빠른 웹 프레임워크
- **LangChain**: LLM 애플리케이션 개발 프레임워크
- **FAISS**: 효율적인 벡터 검색 라이브러리
- **Sentence Transformers**: 텍스트 임베딩 모델
- **OpenAI**: 선택적 LLM 제공자

## 🚀 빠른 시작

### 자동화 스크립트 사용 (권장)

```bash
# 1. 가상환경 설정 및 패키지 설치
./setup.sh

# 2. 가상환경 활성화
./activate.sh

# 3. 서버 실행 (가상환경 활성화 + 서버 시작)
./start.sh

# 4. 정리 (필요시)
./clean.sh
```

### 수동 설정

#### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

#### 2. 환경 변수 설정

```bash
cp env.example .env
```

`.env` 파일을 편집하여 OpenAI API 키를 설정하세요 (선택사항):

```env
OPENAI_API_KEY=your_openai_api_key_here
```

#### 3. 서버 실행

```bash
python run.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

## 📋 스크립트 설명

| 스크립트        | 설명                         |
| --------------- | ---------------------------- |
| `./setup.sh`    | 가상환경 생성 및 패키지 설치 |
| `./activate.sh` | 가상환경 활성화              |
| `./start.sh`    | 가상환경 활성화 + 서버 실행  |
| `./clean.sh`    | 가상환경 및 캐시 파일 정리   |

## API 엔드포인트

### 1. 채팅

**POST** `/chat`

사용자 메시지에 대한 RAG 기반 응답을 생성합니다.

```json
{
  "message": "인공지능이란 무엇인가요?",
  "user_id": "user123"
}
```

응답:

```json
{
  "response": "인공지능(AI)은 컴퓨터가 인간의 지능을 모방하여...",
  "sources": ["sample.txt"],
  "confidence": 0.85
}
```

### 2. 문서 업로드

**POST** `/upload-document`

문서를 업로드하여 벡터 데이터베이스에 저장합니다.

```bash
curl -X POST "http://localhost:8000/upload-document" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.txt"
```

### 3. 문서 목록 조회

**GET** `/documents`

저장된 문서 목록을 반환합니다.

### 4. 문서 삭제

**DELETE** `/documents/{document_id}`

특정 문서를 삭제합니다.

## 테스트

API 테스트를 실행하려면:

```bash
python test_api.py
```

## 사용 예시

### 1. 문서 업로드

```python
import requests

# 문서 업로드
with open('example_documents/sample.txt', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/upload-document', files=files)
    print(response.json())
```

### 2. 채팅

```python
import requests

# 채팅 요청
data = {
    "message": "머신러닝의 종류는 무엇인가요?",
    "user_id": "user123"
}
response = requests.post('http://localhost:8000/chat', json=data)
result = response.json()

print(f"응답: {result['response']}")
print(f"소스: {result['sources']}")
print(f"신뢰도: {result['confidence']}")
```

## 프로젝트 구조

```
simple-chatbot/
├── app.py                 # FastAPI 메인 애플리케이션
├── rag_system.py          # RAG 시스템 핵심 로직
├── run.py                 # 서버 실행 스크립트
├── requirements.txt       # Python 의존성
├── env.example           # 환경 변수 예제
├── test_api.py           # API 테스트 스크립트
├── setup.sh              # 가상환경 설정 스크립트
├── activate.sh            # 가상환경 활성화 스크립트
├── start.sh              # 프로젝트 시작 스크립트
├── clean.sh              # 정리 스크립트
├── static/               # 웹 인터페이스
│   └── index.html
├── example_documents/    # 샘플 문서
│   └── sample.txt
└── README.md
```

## RAG 시스템 작동 원리

1. **문서 처리**: 업로드된 문서를 청크로 분할
2. **벡터화**: 각 청크를 임베딩 모델로 벡터화
3. **검색**: 사용자 질문과 유사한 문서 청크 검색
4. **생성**: 검색된 컨텍스트를 바탕으로 응답 생성
5. **소스 추적**: 응답의 출처 문서 정보 제공

## 접속 정보

- **웹 인터페이스**: http://localhost:8000/web
- **API 문서**: http://localhost:8000/docs
- **API 루트**: http://localhost:8000

## 라이센스

MIT License
