from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from langchain_core.runnables import RunnableConfig
from uuid import uuid4
import json
import os
import httpx
import logging
from typing import Optional
from datetime import datetime
from urllib.parse import unquote

from app.models.schemas import ChatRequest, Session, Message, DocumentReference
from app.services.graph_service import graph_service
from app.api.sessions import sessions_store

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30.0  # 프록시 요청 타임아웃


@router.get("/documents/proxy")
async def proxy_secure_document(url: str):
    """
    보안 문서를 프록시하여 헤더 정보를 포함하여 반환

    Args:
        url: 인코딩된 문서 URL (쿼리 파라미터)

    Returns:
        문서 내용 (PDF, 이미지 등)
    """
    # URL 디코딩
    try:
        decoded_url = unquote(url)
    except Exception as e:
        logger.error(f"URL 디코딩 오류: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid URL: {e}")

    # 환경변수에서 토큰 가져오기
    api_token = os.getenv("EXTERNAL_API_TOKEN", "")

    # 헤더 설정
    headers = {
        "User-Agent": "RAG-Chatbot-Proxy/1.0",
    }

    # 토큰이 있으면 Authorization 헤더 추가
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    try:
        # 실제 문서 URL로 요청
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(
                decoded_url, headers=headers, follow_redirects=True
            )

            if response.status_code != 200:
                logger.error(
                    f"문서 가져오기 실패: status={response.status_code}, url={decoded_url[:100]}"
                )
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch document: {response.status_code}",
                )

            # Content-Type 확인 및 설정
            content_type = response.headers.get(
                "Content-Type", "application/octet-stream"
            )

            # 응답 반환
            return Response(
                content=response.content,
                media_type=content_type,
                headers={
                    "Content-Disposition": 'inline; filename="document"',
                    "Cache-Control": "private, no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

    except httpx.TimeoutException:
        logger.error(f"프록시 요청 타임아웃: url={decoded_url[:100]}")
        raise HTTPException(status_code=504, detail="Request timeout")
    except httpx.RequestError as e:
        logger.error(f"프록시 요청 오류: {e}, url={decoded_url[:100]}")
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch document: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"프록시 엔드포인트 예상치 못한 오류: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/stream")
async def stream_chat(request: ChatRequest):
    """
    스트리밍 채팅 요청 (Server-Sent Events)

    Args:
        request: ChatRequest (message, session_id)

    Returns:
        StreamingResponse: Server-Sent Events 스트리밍 응답
    """
    # session_id가 없으면 새로 생성
    is_new_session = request.session_id is None
    session_id = request.session_id or str(uuid4())

    # 새 세션이면 sessions_store에 저장
    if is_new_session:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        # 첫 메시지의 일부를 제목으로 사용 (나중에 LLM으로 생성 가능)
        title = (
            request.message[:20] + "..."
            if len(request.message) > 20
            else request.message
        )
        session = Session(
            session_id=session_id,
            title=title,
            messages=[],
            created_at=now,
            updated_at=now,
        )
        sessions_store[session_id] = session

    # RunnableConfig 설정 (thread_id로 세션 ID 전달)
    config = RunnableConfig(recursion_limit=10, configurable={"thread_id": session_id})

    async def generate():
        """Server-Sent Events 생성기"""
        full_content = ""
        stream_success = False

        try:
            # 세션 ID 전송
            yield f"data: {json.dumps({'type': 'session_id', 'session_id': session_id})}\n\n"

            # 스트리밍 시작 알림
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            # 메시지 내용 스트리밍
            try:
                async for chunk in graph_service.stream(request.message, config):
                    full_content += chunk
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                stream_success = True
            except Exception as stream_error:
                # OpenSearch 연결 오류 등 스트리밍 중 발생한 오류 처리
                error_type = type(stream_error).__name__
                error_msg = str(stream_error)

                # OpenSearch 연결 오류인 경우 더 명확한 메시지 제공
                if "Connection refused" in error_msg or "ConnectionError" in error_type:
                    error_msg = "OpenSearch 서버에 연결할 수 없습니다. OpenSearch 서버가 실행 중인지 확인해주세요."

                yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                return

            # 스트리밍 완료 알림
            if stream_success:
                # 최종 state에서 api_documents 가져오기
                api_documents = []
                try:
                    final_state = graph_service.get_state(config)
                    api_docs_raw = final_state.get("api_documents", [])

                    # DocumentReference 형식으로 변환
                    if api_docs_raw:
                        api_documents = [
                            DocumentReference(
                                filename=doc.get("filename", ""),
                                content=doc.get("content", ""),
                                url=doc.get("url", ""),
                            )
                            for doc in api_docs_raw
                        ]
                except Exception as e:
                    print(f"문서 정보 가져오기 실패: {e}")

                # 문서 정보가 있으면 전송
                if api_documents:
                    yield f"data: {json.dumps({'type': 'documents', 'documents': [doc.dict() for doc in api_documents]})}\n\n"

                yield f"data: {json.dumps({'type': 'done', 'content': full_content})}\n\n"

                # 세션에 메시지 추가 및 업데이트
                if session_id in sessions_store:
                    user_message = Message(role="user", content=request.message)
                    assistant_message = Message(role="assistant", content=full_content)
                    sessions_store[session_id].messages.append(user_message)
                    sessions_store[session_id].messages.append(assistant_message)
                    sessions_store[session_id].updated_at = datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    )

        except Exception as e:
            # 에러 발생 시
            error_msg = str(e)
            # OpenSearch 연결 오류인 경우 더 명확한 메시지 제공
            if (
                "Connection refused" in error_msg
                or "ConnectionError" in type(e).__name__
            ):
                error_msg = "OpenSearch 서버에 연결할 수 없습니다. OpenSearch 서버가 실행 중인지 확인해주세요."
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 방지
        },
    )
