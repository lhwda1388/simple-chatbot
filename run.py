#!/usr/bin/env python3
"""
RAG 챗봇 API 서버 실행 스크립트
"""

import uvicorn
import os
from dotenv import load_dotenv

def main():
    # 환경 변수 로드
    load_dotenv()
    
    # 서버 설정
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print("🚀 RAG 챗봇 API 서버를 시작합니다...")
    print(f"📍 서버 주소: http://{host}:{port}")
    print(f"🌐 웹 인터페이스: http://{host}:{port}/web")
    print(f"📚 API 문서: http://{host}:{port}/docs")
    print("=" * 50)
    
    # 서버 실행
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 