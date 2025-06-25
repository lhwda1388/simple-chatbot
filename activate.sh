#!/bin/bash

# RAG 챗봇 프로젝트 가상환경 활성화 스크립트

echo "🔧 RAG 챗봇 가상환경을 활성화합니다..."

# 가상환경이 존재하는지 확인
if [ ! -d "venv" ]; then
  echo "❌ 가상환경이 존재하지 않습니다."
  echo "💡 다음 명령어로 가상환경을 생성하세요:"
  echo "   ./setup.sh"
  exit 1
fi

# 가상환경 활성화
source venv/bin/activate

# 활성화 확인
if [ $? -eq 0 ]; then
  echo "✅ 가상환경이 활성화되었습니다!"
  echo ""
  echo "📋 사용 가능한 명령어:"
  echo "   python run.py          - 서버 실행"
  echo "   python test_api.py     - API 테스트"
  echo "   deactivate             - 가상환경 비활성화"
  echo ""
  echo "🌐 웹 인터페이스: http://localhost:8000/web"
  echo "📚 API 문서: http://localhost:8000/docs"
  echo ""
  echo "🎯 서버를 시작하려면: python run.py"

  # 새로운 bash 세션 시작 (가상환경이 활성화된 상태로)
  exec bash
else
  echo "❌ 가상환경 활성화에 실패했습니다."
  exit 1
fi
