#!/bin/bash

echo "🚀 FastAPI CLI로 로컬 LLM 기반 회사 가이드 챗봇 API 서버를 시작합니다..."
echo "=" * 60

# 가상환경 활성화 확인
if [[ "$VIRTUAL_ENV" == "" ]]; then
  echo "📋 가상환경을 활성화합니다..."
  source venv/bin/activate
fi

# Python 버전 확인
echo "🐍 Python 버전 확인:"
python --version

# 필요한 디렉토리 생성
echo "📁 필요한 디렉토리 생성..."
mkdir -p static
mkdir -p example_documents
mkdir -p logs

echo "🔧 FastAPI CLI 실행 설정:"
echo "   - 호스트: 0.0.0.0"
echo "   - 포트: 8000"
echo "   - 웹 인터페이스: http://localhost:8000"
echo "   - API 문서: http://localhost:8000/docs"
echo "   - 재시작: 자동 (--reload)"
echo "=" * 60

# FastAPI CLI로 서버 실행
echo "🚀 서버를 시작합니다..."
echo "   중지하려면 Ctrl+C를 누르세요."
echo "=" * 60

# FastAPI CLI 명령어로 실행
uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level info
