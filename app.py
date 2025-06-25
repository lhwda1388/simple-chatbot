from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from rag_system import RAGSystem
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

app = FastAPI(title="RAG 챗봇 API", description="LangChain 기반 RAG 챗봇 시스템")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# RAG 시스템 초기화
rag_system = RAGSystem()

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    confidence: float

class DocumentUploadResponse(BaseModel):
    message: str
    document_count: int

@app.get("/")
async def root():
    return {"message": "RAG 챗봇 API에 오신 것을 환영합니다!", "web_interface": "/static/index.html"}

@app.get("/web")
async def web_interface():
    """웹 인터페이스로 리다이렉트"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    사용자 메시지에 대해 RAG 기반 응답을 생성합니다.
    """
    try:
        response, sources, confidence = await rag_system.get_response(
            request.message, request.user_id
        )
        return ChatResponse(
            response=response,
            sources=sources,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    문서를 업로드하여 벡터 데이터베이스에 저장합니다.
    """
    try:
        if not file.filename.endswith(('.txt', '.pdf', '.docx', '.md')):
            raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다.")
        
        content = await file.read()
        document_count = await rag_system.add_document(content.decode('utf-8'), file.filename)
        
        return DocumentUploadResponse(
            message="문서가 성공적으로 업로드되었습니다.",
            document_count=document_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 업로드 중 오류가 발생했습니다: {str(e)}")

@app.get("/documents")
async def get_documents():
    """
    저장된 문서 목록을 반환합니다.
    """
    try:
        documents = await rag_system.get_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 목록 조회 중 오류가 발생했습니다: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    특정 문서를 삭제합니다.
    """
    try:
        success = await rag_system.delete_document(document_id)
        if success:
            return {"message": "문서가 성공적으로 삭제되었습니다."}
        else:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 삭제 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 