import os
import asyncio
from typing import List, Tuple, Optional, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_core.documents import Document
import uuid
import json
from datetime import datetime

class RAGSystem:
    def __init__(self):
        """RAG 시스템을 초기화합니다."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # 벡터 스토어 초기화
        self.vector_store = None
        self.documents = {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # LLM 설정 (OpenAI API 키가 있는 경우)
        self.llm = None
        if os.getenv("OPENAI_API_KEY"):
            self.llm = OpenAI(
                temperature=0.7,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # 벡터 스토어 로드 또는 생성
        self._load_or_create_vector_store()
    
    def _load_or_create_vector_store(self):
        """기존 벡터 스토어를 로드하거나 새로 생성합니다."""
        try:
            if os.path.exists("vector_store"):
                self.vector_store = FAISS.load_local("vector_store", self.embeddings)
                self._load_documents_metadata()
                print("기존 벡터 스토어를 로드했습니다.")
            else:
                # 빈 벡터 스토어 생성
                self.vector_store = FAISS.from_texts(
                    ["초기화 텍스트"], 
                    self.embeddings
                )
                self._save_vector_store()
                print("새로운 벡터 스토어를 생성했습니다.")
        except Exception as e:
            print(f"벡터 스토어 로드 중 오류: {e}")
            # 오류 발생 시 새로 생성
            self.vector_store = FAISS.from_texts(
                ["초기화 텍스트"], 
                self.embeddings
            )
            self._save_vector_store()
    
    def _save_vector_store(self):
        """벡터 스토어를 로컬에 저장합니다."""
        if self.vector_store:
            self.vector_store.save_local("vector_store")
            self._save_documents_metadata()
    
    def _save_documents_metadata(self):
        """문서 메타데이터를 저장합니다."""
        with open("documents_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def _load_documents_metadata(self):
        """문서 메타데이터를 로드합니다."""
        try:
            if os.path.exists("documents_metadata.json"):
                with open("documents_metadata.json", "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
        except Exception as e:
            print(f"문서 메타데이터 로드 중 오류: {e}")
            self.documents = {}
    
    async def add_document(self, content: str, filename: str) -> int:
        """문서를 벡터 데이터베이스에 추가합니다."""
        try:
            # 문서 ID 생성
            doc_id = str(uuid.uuid4())
            
            # 텍스트 분할
            chunks = self.text_splitter.split_text(content)
            
            # 문서 메타데이터 저장
            self.documents[doc_id] = {
                "filename": filename,
                "chunks_count": len(chunks),
                "upload_time": datetime.now().isoformat(),
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            }
            
            # 벡터 스토어에 추가
            documents = [Document(page_content=chunk, metadata={"source": filename, "doc_id": doc_id}) 
                        for chunk in chunks]
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)
            
            # 저장
            self._save_vector_store()
            
            return len(self.documents)
            
        except Exception as e:
            print(f"문서 추가 중 오류: {e}")
            raise e
    
    async def get_response(self, query: str, user_id: Optional[str] = None) -> Tuple[str, List[str], float]:
        """사용자 쿼리에 대한 응답을 생성합니다."""
        try:
            if self.vector_store is None:
                return "아직 학습된 문서가 없습니다. 먼저 문서를 업로드해주세요.", [], 0.0
            
            # 유사한 문서 검색
            docs = self.vector_store.similarity_search(query, k=3)
            
            # 소스 문서 추출
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
            
            # 컨텍스트 구성
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # LLM이 있는 경우 RAG 응답 생성
            if self.llm:
                prompt = f"""
                다음 컨텍스트를 바탕으로 사용자의 질문에 답변해주세요.
                
                컨텍스트:
                {context}
                
                사용자 질문: {query}
                
                답변:
                """
                
                response = self.llm(prompt)
                confidence = 0.8  # 기본 신뢰도
                
            else:
                # LLM이 없는 경우 간단한 키워드 매칭 응답
                response = self._simple_response(query, context)
                confidence = 0.6
            
            return response, sources, confidence
            
        except Exception as e:
            print(f"응답 생성 중 오류: {e}")
            return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다.", [], 0.0
    
    def _simple_response(self, query: str, context: str) -> str:
        """LLM이 없을 때 사용하는 간단한 응답 생성"""
        # 키워드 기반 간단한 응답
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["무엇", "뭐", "어떤"]):
            return f"문서에서 찾은 정보: {context[:300]}..."
        elif any(word in query_lower for word in ["어떻게", "방법", "절차"]):
            return f"절차에 대한 정보: {context[:300]}..."
        else:
            return f"관련 정보: {context[:300]}..."
    
    async def get_documents(self) -> List[Dict[str, Any]]:
        """저장된 문서 목록을 반환합니다."""
        return [
            {
                "id": doc_id,
                "filename": metadata["filename"],
                "chunks_count": metadata["chunks_count"],
                "upload_time": metadata["upload_time"],
                "content_preview": metadata["content_preview"]
            }
            for doc_id, metadata in self.documents.items()
        ]
    
    async def delete_document(self, document_id: str) -> bool:
        """특정 문서를 삭제합니다."""
        try:
            if document_id not in self.documents:
                return False
            
            # 해당 문서의 청크들을 벡터 스토어에서 제거
            # FAISS는 개별 문서 삭제를 직접 지원하지 않으므로 재구성
            all_docs = []
            for doc_id, metadata in self.documents.items():
                if doc_id != document_id:
                    # 해당 문서의 청크들을 다시 추가
                    # 실제 구현에서는 더 정교한 방법이 필요
                    pass
            
            # 문서 메타데이터에서 제거
            del self.documents[document_id]
            
            # 저장
            self._save_vector_store()
            
            return True
            
        except Exception as e:
            print(f"문서 삭제 중 오류: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보를 반환합니다."""
        return {
            "total_documents": len(self.documents),
            "vector_store_exists": self.vector_store is not None,
            "llm_available": self.llm is not None,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        } 