from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
from rag_system import RAGSystem
import json

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="로컬 LLM 기반 회사 가이드 챗봇 API", 
    version="1.0.0",
    description="RAG 기술을 활용한 지능형 문서 기반 질의응답 시스템"
)

# 정적 파일 서빙 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# RAG 시스템 초기화
rag_system = RAGSystem()

# Pydantic 모델 정의
class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str
    user_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    response: str
    sources: List[str]
    confidence: float
    model_used: str

class DocumentResponse(BaseModel):
    """문서 업로드 응답 모델"""
    document_id: str
    filename: str
    chunks: int
    status: str

# API 엔드포인트 정의
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """웹 인터페이스 제공"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 API - RAG 기반 응답 생성"""
    try:
        response, sources, confidence, model_used = await rag_system.chat(
            request.message, request.user_id or ""
        )
        return ChatResponse(
            response=response,
            sources=sources,
            confidence=confidence,
            model_used=model_used
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류 발생: {str(e)}")

@app.post("/upload-document", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """문서 업로드 및 벡터화"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다")
        
        # 파일 확장자 검사
        allowed_extensions = {'.txt', '.md', '.pdf', '.docx'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {allowed_extensions}"
            )
        
        # 파일 내용 읽기
        content = await file.read()
        if file_ext == '.txt' or file_ext == '.md':
            text_content = content.decode('utf-8')
        else:
            # PDF, DOCX 등은 간단한 텍스트 추출 (실제로는 더 정교한 라이브러리 필요)
            text_content = content.decode('utf-8', errors='ignore')
        
        # RAG 시스템에 문서 추가
        document_id = await rag_system.add_document(file.filename, text_content)
        
        return DocumentResponse(
            document_id=document_id,
            filename=file.filename,
            chunks=len(text_content.split('\n')),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 업로드 중 오류 발생: {str(e)}")

@app.get("/documents")
async def list_documents():
    """저장된 문서 목록 조회"""
    try:
        documents = await rag_system.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 목록 조회 중 오류 발생: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """문서 삭제"""
    try:
        success = await rag_system.delete_document(document_id)
        if success:
            return {"message": f"문서 {document_id}가 삭제되었습니다"}
        else:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 삭제 중 오류 발생: {str(e)}")

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_status": rag_system.get_model_status(),
        "vector_db_status": "ready"
    }
