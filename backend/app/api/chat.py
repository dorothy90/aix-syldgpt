from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from uuid import uuid4
import json
from typing import Optional
from datetime import datetime
import logging

from app.models.schemas import ChatRequest, Session, Message
from app.services.graph_service import graph_service
from app.api.sessions import sessions_store

# 디버그 로거 설정
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
    # ==================== 디버그 로그 ====================
    logger.debug("=" * 60)
    logger.debug("[SESSION DEBUG] 새 요청 수신")
    logger.debug(f"[SESSION DEBUG] 요청 메시지: {request.message[:50]}...")
    logger.debug(f"[SESSION DEBUG] 요청된 session_id: {request.session_id}")
    logger.debug(f"[SESSION DEBUG] session_id 타입: {type(request.session_id)}")
    logger.debug(f"[SESSION DEBUG] session_id is None: {request.session_id is None}")
    logger.debug(f"[SESSION DEBUG] session_id == '': {request.session_id == ''}")
    logger.debug(
        f"[SESSION DEBUG] 현재 sessions_store 키: {list(sessions_store.keys())}"
    )
    # =====================================================

    # session_id가 없으면 새로 생성
    is_new_session = request.session_id is None
    session_id = request.session_id or str(uuid4())

    # ==================== 디버그 로그 ====================
    logger.debug(f"[SESSION DEBUG] is_new_session 판정: {is_new_session}")
    logger.debug(f"[SESSION DEBUG] 사용할 session_id: {session_id}")
    # =====================================================

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
        # ==================== 디버그 로그 ====================
        logger.info(f"[SESSION DEBUG] ⭐ 새 세션 생성됨: {session_id}")
        logger.debug(f"[SESSION DEBUG] 세션 제목: {title}")
        # =====================================================
    else:
        # ==================== 디버그 로그 ====================
        if session_id in sessions_store:
            existing_session = sessions_store[session_id]
            logger.info(f"[SESSION DEBUG] ✅ 기존 세션 사용: {session_id}")
            logger.debug(
                f"[SESSION DEBUG] 기존 세션 메시지 수: {len(existing_session.messages)}"
            )
        else:
            logger.warning(
                f"[SESSION DEBUG] ⚠️ session_id가 있지만 sessions_store에 없음: {session_id}"
            )
            logger.warning(
                f"[SESSION DEBUG] 현재 store 키들: {list(sessions_store.keys())}"
            )
        # =====================================================

    # RunnableConfig 설정 (thread_id로 세션 ID 전달)
    config = RunnableConfig(recursion_limit=10, configurable={"thread_id": session_id})

    async def generate():
        """Server-Sent Events 생성기"""
        full_content = ""
        stream_success = False

        try:
            # 세션 ID 전송
            logger.debug(f"[SESSION DEBUG] 클라이언트에 session_id 전송: {session_id}")
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
                # 그래프 최종 상태에서 answer/artifacts를 회수
                final_answer = full_content
                artifacts = []
                try:
                    final_state = graph_service.app.get_state(config)
                    values = getattr(final_state, "values", {}) or {}
                    if values.get("answer"):
                        final_answer = values["answer"]
                    artifacts = values.get("artifacts") or []
                except Exception:
                    # 상태 조회 실패 시에도 기존 스트리밍 결과는 유지
                    artifacts = []

                yield f"data: {json.dumps({'type': 'done', 'content': final_answer})}\n\n"
                if artifacts:
                    yield f"data: {json.dumps({'type': 'artifacts', 'artifacts': artifacts})}\n\n"

                # 세션에 메시지 추가 및 업데이트
                if session_id in sessions_store:
                    user_message = Message(role="user", content=request.message)
                    assistant_message = Message(
                        role="assistant",
                        content=final_answer,
                        artifacts=artifacts or None,
                    )
                    sessions_store[session_id].messages.append(user_message)
                    sessions_store[session_id].messages.append(assistant_message)
                    sessions_store[session_id].updated_at = datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    # ==================== 디버그 로그 ====================
                    logger.debug(f"[SESSION DEBUG] 세션에 메시지 저장 완료")
                    logger.debug(
                        f"[SESSION DEBUG] 현재 세션 메시지 수: {len(sessions_store[session_id].messages)}"
                    )
                    logger.debug("=" * 60)
                    # =====================================================

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
