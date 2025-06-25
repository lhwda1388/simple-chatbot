#!/bin/bash

# RAG 챗봇 프로젝트 가상환경 설정 스크립트

echo "🚀 RAG 챗봇 프로젝트 가상환경 설정을 시작합니다..."

# 프로젝트 디렉토리 확인
if [ ! -f "requirements.txt" ]; then
  echo "❌ requirements.txt 파일을 찾을 수 없습니다. 프로젝트 루트 디렉토리에서 실행해주세요."
  exit 1
fi

# Python 버전 확인
echo "🐍 Python 버전 확인 중..."
python3 --version

# 기존 가상환경이 있는지 확인
if [ -d "venv" ]; then
  echo "⚠️  기존 가상환경이 발견되었습니다."
  read -p "기존 가상환경을 삭제하고 새로 생성하시겠습니까? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  기존 가상환경 삭제 중..."
    rm -rf venv
  else
    echo "❌ 설정을 취소했습니다."
    exit 1
  fi
fi

# 가상환경 생성
echo "📦 가상환경 생성 중..."
python3 -m venv venv

# 가상환경 활성화
echo "🔧 가상환경 활성화 중..."
source venv/bin/activate

# pip 업그레이드
echo "⬆️  pip 업그레이드 중..."
pip install --upgrade pip

# 의존성 설치
echo "📚 필요한 패키지 설치 중..."
pip install -r requirements.txt

# 설치 완료 확인
if [ $? -eq 0 ]; then
  echo ""
  echo "✅ 가상환경 설정이 완료되었습니다!"
  echo ""
  echo "📋 사용 방법:"
  echo "   가상환경 활성화: source venv/bin/activate"
  echo "   서버 실행: python run.py"
  echo "   가상환경 비활성화: deactivate"
  echo ""
  echo "🌐 웹 인터페이스: http://localhost:8000/web"
  echo "📚 API 문서: http://localhost:8000/docs"
  echo ""
  echo "🎯 다음 명령어로 서버를 시작할 수 있습니다:"
  echo "   python run.py"
else
  echo "❌ 패키지 설치 중 오류가 발생했습니다."
  exit 1
fi
