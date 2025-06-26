import asyncio
import json
import os
import uuid
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch



BASIC_GUIDE = """
당신은 회사 가이드 챗봇입니다.
아래 회사의 규정과 안내를 참고하여, 사용자의 질문에 친절하고 구체적으로 답변하세요.
"""

VECTOR_STORE_PATH = "vector_store"


class RAGSystem:
    """
    RAG(Retrieval-Augmented Generation) 시스템 클래스
    
    문서 기반 질의응답을 위한 벡터 데이터베이스와 언어 모델을 관리합니다.
    """
    
    def __init__(self) -> None:
        """
        RAG 시스템 초기화
        """
        self.vector_store: Optional[FAISS] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.llm: Optional[HuggingFacePipeline] = None
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.model_name: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B"  # 가벼운 한국어 지원 모델
        
        # 초기화
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """
        RAG 시스템 컴포넌트 초기화
        
        임베딩 모델, LLM, 벡터 스토어를 초기화합니다.
        """
        try:
            print("임베딩 모델 초기화 중...")
            # 임베딩 모델 초기화 (한국어 지원)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )

            print("임베딩 모델 초기화 완료")
            
            # 로컬 LLM 초기화 (직접 모델 로드)
            self._load_local_llm()

            print("로컬 LLM 모델 초기화 완료")
            
            # 벡터 스토어 초기화
            self._load_or_create_vector_store()
            
            print(f"RAG 시스템 초기화 완료 - 모델: {self.model_name}")
            
        except Exception as e:
            print(f"RAG 시스템 초기화 실패: {str(e)}")
            print("모델 다운로드 중 오류가 발생했습니다. 인터넷 연결을 확인해주세요.")
    
    def _load_local_llm(self) -> None:
        """
        로컬 LLM 모델 로드
        
        기본 모델을 로드하고 실패 시 대체 모델을 시도합니다.
        """
        try:
            print("로컬 LLM 모델을 로드하는 중...")
        
            
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            # 모델 로드 (CPU 사용으로 메모리 절약)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            
            # 파이프라인 생성
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # LangChain LLM 래퍼 생성
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            print("로컬 LLM 모델 로드 완료")
            
        except Exception as e:
            print(f"로컬 LLM 로드 실패: {str(e)}")
    
    def _load_or_create_vector_store(self) -> None:
        """
        벡터 스토어 로드 또는 생성
        
        기존 벡터 스토어가 있으면 로드하고, 없으면 새로 생성합니다.
        """
        try:
            if os.path.exists(VECTOR_STORE_PATH):
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("기존 벡터 스토어를 로드했습니다.")
            else:
                # 빈 벡터 스토어 생성
                dummy_text = "초기화"
                dummy_docs = [Document(page_content=dummy_text, metadata={"source": "init"})]
                self.vector_store = FAISS.from_documents(dummy_docs, self.embeddings)
                self.vector_store.save_local(VECTOR_STORE_PATH)
                print("새로운 벡터 스토어를 생성했습니다.")
        except Exception as e:
            print(f"벡터 스토어 로드 실패: {str(e)}")
    
    async def add_document(self, filename: str, content: str) -> str:
        """
        문서를 벡터 데이터베이스에 추가
        
        Args:
            filename: 문서 파일명
            content: 문서 내용
            
        Returns:
            str: 생성된 문서 ID
            
        Raises:
            Exception: 문서 추가 중 오류 발생 시
        """
        try:
            # 문서 ID 생성
            document_id = str(uuid.uuid4())
            
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_text(content)
            
            # 문서 메타데이터 생성
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "document_id": document_id,
                        "chunk_id": i,
                        "created_at": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            
            # 벡터 스토어에 추가
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)
            
            # 벡터 스토어 저장
            self.vector_store.save_local(VECTOR_STORE_PATH)
            
            # 문서 정보 저장
            self.documents[document_id] = {
                "filename": filename,
                "chunks": len(chunks),
                "created_at": datetime.now().isoformat(),
                "content_length": len(content)
            }
            
            # 문서 정보를 파일에 저장
            self._save_documents_info()
            
            print(f"문서 '{filename}' 추가 완료 - {len(chunks)}개 청크")
            return document_id
            
        except Exception as e:
            print(f"문서 추가 실패: {str(e)}")
            raise e
    
    async def chat(self, message: str, user_id: str = "") -> Tuple[str, List[str], float, str]:
        """
        RAG 기반 채팅 응답 생성
        
        Args:
            message: 사용자 메시지
            user_id: 사용자 ID (선택 사항)
            
        Returns:
            Tuple[str, List[str], float, str]: (응답 텍스트, 참조 소스 목록, 신뢰도, 모델명)
            
        Raises:
            Exception: 채팅 처리 중 오류 발생 시
        """
        try:
            if self.vector_store is None:
                return "벡터 데이터베이스가 초기화되지 않았습니다.", [], 0.0, "error"
            
            if self.llm is None:
                return "LLM 모델이 로드되지 않았습니다. 모델 다운로드를 확인해주세요.", [], 0.0, "error"
            
            # 유사한 문서 검색
            docs = self.vector_store.similarity_search(message, k=3)
            
            if not docs:
                return "관련된 문서를 찾을 수 없습니다.", [], 0.0, self.model_name
            
            # 컨텍스트 구성
            context = "\n\n".join([doc.page_content for doc in docs])
            sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
            
            # 프롬프트 템플릿 (더 간단한 형태로 변경)
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=f"""{BASIC_GUIDE}

[참고 문서]
{{context}}

[질문]
{{question}}

[답변]
"""
            )
            
            # LLM에 질문 (동기 방식으로 변경)
            formatted_prompt = prompt_template.format(context=context, question=message)
            
            # 모델 응답 생성
            response = self.llm(formatted_prompt)
            
            # 응답 정리 (첫 번째 문장만 추출)
            if response:
                response_text = response.strip()
                # 첫 번째 완전한 문장만 추출
                sentences = response_text.split('.')
                if sentences:
                    response_text = sentences[0].strip() + '.'
            else:
                response_text = "응답을 생성할 수 없습니다."
            
            # 신뢰도 계산 (간단한 휴리스틱)
            confidence = min(0.9, 0.5 + len(context) / 10000)
            
            return response_text, sources, confidence, self.model_name
            
        except Exception as e:
            print(f"채팅 처리 실패: {str(e)}")
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}", [], 0.0, self.model_name
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        저장된 문서 목록 반환
        
        Returns:
            List[Dict[str, Any]]: 문서 정보 목록
        """
        try:
            self._load_documents_info()
            return [
                {
                    "document_id": doc_id,
                    **doc_info
                }
                for doc_id, doc_info in self.documents.items()
            ]
        except Exception as e:
            print(f"문서 목록 조회 실패: {str(e)}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """
        문서 삭제
        
        Args:
            document_id: 삭제할 문서 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if document_id not in self.documents:
                return False
            
            # 벡터 스토어에서 해당 문서의 청크들 제거
            if self.vector_store is not None:
                # 현재 벡터 스토어를 다시 로드하여 최신 상태 유지
                self.vector_store = FAISS.load_local(VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True)
                
                # 해당 문서의 청크들을 제외하고 새로운 벡터 스토어 생성
                all_docs = self.vector_store.docstore._dict.values()
                filtered_docs = [
                    doc for doc in all_docs 
                    if doc.metadata.get("document_id") != document_id
                ]
                
                if filtered_docs:
                    self.vector_store = FAISS.from_documents(filtered_docs, self.embeddings)
                else:
                    # 모든 문서가 삭제된 경우 빈 벡터 스토어 생성
                    dummy_docs = [Document(page_content="초기화", metadata={"source": "init"})]
                    self.vector_store = FAISS.from_documents(dummy_docs, self.embeddings)
                
                    self.vector_store.save_local(VECTOR_STORE_PATH)
            
            # 문서 정보에서 제거
            del self.documents[document_id]
            self._save_documents_info()
            
            print(f"문서 {document_id} 삭제 완료")
            return True
            
        except Exception as e:
            print(f"문서 삭제 실패: {str(e)}")
            return False
    
    def get_model_status(self) -> str:
        """
        모델 상태 확인
        
        Returns:
            str: 모델 상태 ('ready', 'not_loaded', 'error')
        """
        try:
            if self.llm is None:
                return "not_loaded"
            
            # 간단한 테스트 요청
            test_response = self.llm("Hello")
            if test_response:
                return "ready"
            else:
                return "error"
        except Exception as e:
            return f"error: {str(e)}"
    
    def _save_documents_info(self) -> None:
        """
        문서 정보를 파일에 저장
        """
        try:
            with open("documents_info.json", "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"문서 정보 저장 실패: {str(e)}")
    
    def _load_documents_info(self) -> None:
        """
        문서 정보를 파일에서 로드
        """
        try:
            if os.path.exists("documents_info.json"):
                with open("documents_info.json", "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
        except Exception as e:
            print(f"문서 정보 로드 실패: {str(e)}")
            self.documents = {}