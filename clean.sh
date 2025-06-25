#!/bin/bash

# RAG 챗봇 프로젝트 정리 스크립트

echo "🧹 RAG 챗봇 프로젝트 정리를 시작합니다..."

# 현재 가상환경이 활성화되어 있는지 확인
if [[ "$VIRTUAL_ENV" != "" ]]; then
  echo "⚠️  가상환경이 활성화되어 있습니다. 비활성화합니다..."
  deactivate
fi

# 정리할 항목들
echo "🗑️  다음 항목들을 정리합니다:"

# 가상환경 삭제
if [ -d "venv" ]; then
  echo "   - 가상환경 (venv/)"
  rm -rf venv
fi

# Python 캐시 파일들 삭제
echo "   - Python 캐시 파일들"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# 벡터 스토어 삭제
if [ -d "vector_store" ]; then
  echo "   - 벡터 스토어 (vector_store/)"
  rm -rf vector_store
fi

# 문서 메타데이터 삭제
if [ -f "documents_metadata.json" ]; then
  echo "   - 문서 메타데이터 (documents_metadata.json)"
  rm -f documents_metadata.json
fi

# .env 파일 삭제 (백업 생성)
if [ -f ".env" ]; then
  echo "   - 환경 변수 파일 (.env -> .env.backup)"
  mv .env .env.backup
fi

echo ""
echo "✅ 정리가 완료되었습니다!"
echo ""
echo "💡 다음 명령어로 프로젝트를 다시 설정할 수 있습니다:"
echo "   ./setup.sh"
