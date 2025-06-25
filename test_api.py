#!/usr/bin/env python3
"""
로컬 LLM 기반 회사 가이드 챗봇 API 테스트 스크립트
"""

import requests
import json
import time
import os
import uuid
from datetime import datetime
from typing import List, Tuple, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

BASE_URL = "http://localhost:8000"

def test_health():
    """헬스 체크 테스트"""
    print("🔍 헬스 체크 테스트...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 서버 상태: {data['status']}")
            print(f"✅ 모델 상태: {data['model_status']}")
            print(f"✅ 벡터 DB 상태: {data['vector_db_status']}")
            return True
        else:
            print(f"❌ 헬스 체크 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 헬스 체크 오류: {e}")
        return False

def test_document_upload():
    """문서 업로드 테스트"""
    print("\n📄 문서 업로드 테스트...")
    try:
        # 샘플 문서 파일 경로
        sample_file = "example_documents/company_guide.txt"
        
        if not os.path.exists(sample_file):
            print(f"❌ 샘플 파일이 없습니다: {sample_file}")
            return False
        
        with open(sample_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/upload-document", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 문서 업로드 성공: {data['filename']}")
            print(f"✅ 문서 ID: {data['document_id']}")
            print(f"✅ 청크 수: {data['chunks']}")
            return data['document_id']
        else:
            print(f"❌ 문서 업로드 실패: {response.status_code}")
            print(f"❌ 오류: {response.text}")
            return None
    except Exception as e:
        print(f"❌ 문서 업로드 오류: {e}")
        return None

def test_document_list():
    """문서 목록 조회 테스트"""
    print("\n📚 문서 목록 조회 테스트...")
    try:
        response = requests.get(f"{BASE_URL}/documents")
        if response.status_code == 200:
            data = response.json()
            documents = data.get('documents', [])
            print(f"✅ 저장된 문서 수: {len(documents)}")
            for doc in documents:
                print(f"   - {doc['filename']} (ID: {doc['document_id']})")
            return True
        else:
            print(f"❌ 문서 목록 조회 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 문서 목록 조회 오류: {e}")
        return False

def test_chat(question):
    """채팅 테스트"""
    print(f"\n💬 채팅 테스트: '{question}'")
    try:
        data = {
            "message": question,
            "user_id": "test_user"
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 응답: {result['response']}")
            print(f"✅ 소스: {result['sources']}")
            print(f"✅ 신뢰도: {result['confidence']:.2f}")
            print(f"✅ 모델: {result['model_used']}")
            return True
        else:
            print(f"❌ 채팅 실패: {response.status_code}")
            print(f"❌ 오류: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 채팅 오류: {e}")
        return False

def test_document_delete(document_id):
    """문서 삭제 테스트"""
    print(f"\n🗑️  문서 삭제 테스트 (ID: {document_id})")
    try:
        response = requests.delete(f"{BASE_URL}/documents/{document_id}")
        
        if response.status_code == 200:
            print("✅ 문서 삭제 성공")
            return True
        else:
            print(f"❌ 문서 삭제 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 문서 삭제 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🧪 로컬 LLM 기반 회사 가이드 챗봇 API 테스트 시작")
    print("=" * 60)
    
    # 서버가 실행 중인지 확인
    print("🔍 서버 연결 확인 중...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print("✅ 서버에 연결되었습니다.")
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다.")
        print("서버를 먼저 실행해주세요: python run.py")
        return
    except Exception as e:
        print(f"❌ 연결 오류: {e}")
        return
    
    # 테스트 실행
    tests_passed = 0
    total_tests = 0
    
    # 1. 헬스 체크 테스트
    total_tests += 1
    if test_health():
        tests_passed += 1
    
    # 2. 문서 업로드 테스트
    total_tests += 1
    document_id = test_document_upload()
    if document_id:
        tests_passed += 1
    
    # 3. 문서 목록 조회 테스트
    total_tests += 1
    if test_document_list():
        tests_passed += 1
    
    # 4. 채팅 테스트들
    test_questions = [
        "회사는 언제 설립되었나요?",
        "근무 시간은 어떻게 되나요?",
        "복리후생에는 무엇이 포함되어 있나요?",
        "개발팀은 몇 명인가요?",
        "보안 정책은 어떻게 되나요?"
    ]
    
    for question in test_questions:
        total_tests += 1
        if test_chat(question):
            tests_passed += 1
        time.sleep(1)  # 요청 간 간격
    
    # 5. 문서 삭제 테스트 (선택적)
    if document_id:
        total_tests += 1
        if test_document_delete(document_id):
            tests_passed += 1
    
    # 결과 출력
    print("\n" + "=" * 60)
    print(f"📊 테스트 결과: {tests_passed}/{total_tests} 통과")
    
    if tests_passed == total_tests:
        print("🎉 모든 테스트가 성공적으로 통과했습니다!")
    else:
        print("⚠️  일부 테스트가 실패했습니다.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 