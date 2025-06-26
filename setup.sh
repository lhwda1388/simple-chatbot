#!/bin/bash

echo "🚀 로컬 LLM 기반 회사 가이드 챗봇 설정을 시작합니다..."
echo "=" * 60

# Python 버전 확인
echo "📋 Python 버전 확인 중..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "현재 Python 버전: $python_version"

# 가상환경 생성
echo "🔧 가상환경 생성 중..."
if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "✅ 가상환경이 생성되었습니다."
else
  echo "ℹ️  가상환경이 이미 존재합니다."
fi

# 가상환경 활성화
echo "🔗 가상환경 활성화 중..."
source venv/bin/activate

# pip 업그레이드
echo "⬆️  pip 업그레이드 중..."
pip install --upgrade pip

# 의존성 설치
echo "📦 필요한 패키지 설치 중..."
pip install -r requirements.txt

# 필요한 디렉토리 생성
echo "📁 필요한 디렉토리 생성 중..."
mkdir -p static
mkdir -p example_documents
mkdir -p logs

echo "✅ 설정이 완료되었습니다!"
echo "=" * 60
echo "다음 단계:"
echo "1. 가상환경 활성화: source venv/bin/activate"
echo "2. 서버 실행: start.sh"
echo "3. 웹 브라우저에서 http://localhost:8000 접속"
echo "=" * 60
