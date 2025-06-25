import requests
import json
import time

# API 기본 URL
BASE_URL = "http://localhost:8000"

def test_root():
    """루트 엔드포인트 테스트"""
    print("=== 루트 엔드포인트 테스트 ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"상태 코드: {response.status_code}")
    print(f"응답: {response.json()}")
    print()

def test_chat():
    """채팅 엔드포인트 테스트"""
    print("=== 채팅 엔드포인트 테스트 ===")
    
    # 테스트 메시지들
    test_messages = [
        "인공지능이란 무엇인가요?",
        "머신러닝의 종류는 무엇이 있나요?",
        "RAG 기술에 대해 설명해주세요."
    ]
    
    for message in test_messages:
        print(f"질문: {message}")
        
        data = {
            "message": message,
            "user_id": "test_user"
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"응답: {result['response']}")
            print(f"소스: {result['sources']}")
            print(f"신뢰도: {result['confidence']}")
        else:
            print(f"오류: {response.status_code} - {response.text}")
        
        print("-" * 50)

def test_upload_document():
    """문서 업로드 테스트"""
    print("=== 문서 업로드 테스트 ===")
    
    # 샘플 문서 내용
    sample_content = """
    파이썬 프로그래밍 언어에 대한 정보
    
    파이썬은 1991년 귀도 반 로섬이 개발한 고급 프로그래밍 언어입니다.
    
    파이썬의 특징:
    1. 간단하고 읽기 쉬운 문법
    2. 풍부한 라이브러리 생태계
    3. 크로스 플랫폼 지원
    4. 동적 타입 언어
    
    파이썬은 웹 개발, 데이터 분석, 인공지능, 자동화 등 다양한 분야에서 사용됩니다.
    
    주요 파이썬 프레임워크:
    - Django: 웹 프레임워크
    - Flask: 경량 웹 프레임워크
    - FastAPI: 현대적인 API 프레임워크
    - Pandas: 데이터 분석 라이브러리
    - NumPy: 수치 계산 라이브러리
    """
    
    # 파일 업로드 시뮬레이션
    files = {
        'file': ('python_info.txt', sample_content, 'text/plain')
    }
    
    response = requests.post(f"{BASE_URL}/upload-document", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"업로드 성공: {result['message']}")
        print(f"문서 수: {result['document_count']}")
    else:
        print(f"업로드 실패: {response.status_code} - {response.text}")
    
    print()

def test_get_documents():
    """문서 목록 조회 테스트"""
    print("=== 문서 목록 조회 테스트 ===")
    
    response = requests.get(f"{BASE_URL}/documents")
    
    if response.status_code == 200:
        result = response.json()
        documents = result.get('documents', [])
        
        if documents:
            print(f"총 {len(documents)}개의 문서가 있습니다:")
            for doc in documents:
                print(f"- ID: {doc['id']}")
                print(f"  파일명: {doc['filename']}")
                print(f"  청크 수: {doc['chunks_count']}")
                print(f"  업로드 시간: {doc['upload_time']}")
                print(f"  미리보기: {doc['content_preview'][:100]}...")
                print()
        else:
            print("저장된 문서가 없습니다.")
    else:
        print(f"조회 실패: {response.status_code} - {response.text}")
    
    print()

def main():
    """메인 테스트 함수"""
    print("RAG 챗봇 API 테스트를 시작합니다...")
    print("=" * 60)
    
    # 서버가 실행 중인지 확인
    try:
        test_root()
    except requests.exceptions.ConnectionError:
        print("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
        print("서버 실행 명령: python app.py")
        return
    
    # 문서 업로드 테스트
    test_upload_document()
    
    # 문서 목록 조회 테스트
    test_get_documents()
    
    # 채팅 테스트
    test_chat()
    
    print("테스트가 완료되었습니다.")

if __name__ == "__main__":
    main() 