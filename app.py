from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
from rag_system import RAGSystem

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


class ChatRequest(BaseModel):
    """채팅 요청 모델
    
    Attributes:
        message: 사용자 메시지
        user_id: 사용자 식별자
    """
    message: str
    user_id: Optional[str] = "default_user"


class ChatResponse(BaseModel):
    """채팅 응답 모델
    
    Attributes:
        response: 생성된 응답 텍스트
        sources: 참조된 문서 소스 목록
        confidence: 응답 신뢰도 점수
        model_used: 사용된 모델 이름
    """
    response: str
    sources: List[str]
    confidence: float
    model_used: str


class DocumentResponse(BaseModel):
    """문서 업로드 응답 모델
    
    Attributes:
        document_id: 문서 고유 식별자
        filename: 업로드된 파일명
        chunks: 분할된 청크 수
        status: 처리 상태
    """
    document_id: str
    filename: str
    chunks: int
    status: str


@app.get("/", response_class=HTMLResponse)
async def read_root() -> HTMLResponse:
    """웹 인터페이스 제공
    
    Returns:
        HTMLResponse: 메인 페이지 HTML 컨텐츠
    """
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"인덱스 페이지 로드 실패: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """채팅 API - RAG 기반 응답 생성
    
    Args:
        request: 채팅 요청 객체
        
    Returns:
        ChatResponse: 생성된 응답 정보
        
    Raises:
        HTTPException: 처리 중 오류 발생 시
    """
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
async def upload_document(file: UploadFile = File(...)) -> DocumentResponse:
    """문서 업로드 및 벡터화
    
    Args:
        file: 업로드된 파일 객체
        
    Returns:
        DocumentResponse: 문서 처리 결과
        
    Raises:
        HTTPException: 파일 처리 중 오류 발생 시
    """
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
        if file_ext in ('.txt', '.md'):
            text_content = content.decode('utf-8')
        else:
            # PDF, DOCX 등은 간단한 텍스트 추출
            text_content = content.decode('utf-8', errors='ignore')
        

        print("text_content", text_content)
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
async def list_documents() -> dict:
    """저장된 문서 목록 조회
    
    Returns:
        dict: 문서 목록 정보
        
    Raises:
        HTTPException: 조회 중 오류 발생 시
    """
    try:
        documents = await rag_system.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 목록 조회 중 오류 발생: {str(e)}")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str) -> dict:
    """문서 삭제
    
    Args:
        document_id: 삭제할 문서 ID
        
    Returns:
        dict: 삭제 결과 메시지
        
    Raises:
        HTTPException: 문서가 존재하지 않거나 삭제 중 오류 발생 시
    """
    try:
        success = await rag_system.delete_document(document_id)
        if success:
            return {"message": f"문서 {document_id}가 삭제되었습니다"}
        else:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 삭제 중 오류 발생: {str(e)}")


@app.get("/health")
async def health_check() -> dict:
    """시스템 상태 확인
    
    Returns:
        dict: 시스템 상태 정보
    """
    try:
        return {
            "status": "healthy",
            "model_status": rag_system.get_model_status(),
            "vector_db_status": "ready" if rag_system.vector_store else "not_initialized"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
