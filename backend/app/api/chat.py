from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from uuid import uuid4
import json
from typing import Optional
from datetime import datetime

from app.models.schemas import ChatRequest, Session, Message
from app.services.graph_service import graph_service
from app.api.sessions import sessions_store

router = APIRouter(prefix="/api/chat", tags=["chat"])


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
