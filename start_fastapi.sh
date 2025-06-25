#!/bin/bash

# FastAPI CLI를 사용한 RAG 챗봇 서버 실행 스크립트

echo "🚀 FastAPI CLI로 RAG 챗봇 서버를 시작합니다..."

# 가상환경이 존재하는지 확인
if [ ! -d "venv" ]; then
  echo "❌ 가상환경이 존재하지 않습니다."
  echo "💡 가상환경을 먼저 생성하세요:"
  echo "   ./setup.sh"
  exit 1
fi

# 가상환경 활성화
echo "🔧 가상환경 활성화 중..."
source venv/bin/activate

if [ $? -ne 0 ]; then
  echo "❌ 가상환경 활성화에 실패했습니다."
  exit 1
fi

echo "✅ 가상환경이 활성화되었습니다!"

# FastAPI CLI 사용
echo "🌐 FastAPI CLI로 서버를 시작합니다..."
echo "📍 서버 주소: http://0.0.0.0:8000"
echo "🌐 웹 인터페이스: http://0.0.0.0:8000/web"
echo "📚 API 문서: http://0.0.0.0:8000/docs"
echo ""
echo "🛑 서버를 중지하려면 Ctrl+C를 누르세요."
echo "=" * 50

# FastAPI CLI 실행 (개발 모드)
python -m fastapi dev app.py --host 0.0.0.0 --port 8000
