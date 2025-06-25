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
else
  echo "❌ 가상환경 활성화에 실패했습니다."
  exit 1
fi
