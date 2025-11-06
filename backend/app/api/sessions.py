from fastapi import APIRouter, HTTPException
from uuid import uuid4
from datetime import datetime
from typing import Dict, List

from app.models.schemas import Session, SessionCreate, Message

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# 메모리 기반 세션 저장소 (향후 MongoDB로 교체 가능)
# 다른 모듈에서도 사용할 수 있도록 export
sessions_store: Dict[str, Session] = {}


@router.post("/", response_model=Session)
async def create_session(session_create: SessionCreate = None):
    """
    새 세션 생성

    Args:
        session_create: SessionCreate (title 선택사항)

    Returns:
        Session: 생성된 세션
    """
    session_id = str(uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    title = (
        session_create.title
        if session_create and session_create.title
        else f"새 채팅 {len(sessions_store) + 1}"
    )

    session = Session(
        session_id=session_id, title=title, messages=[], created_at=now, updated_at=now
    )

    sessions_store[session_id] = session
    return session


@router.get("/", response_model=List[Session])
async def list_sessions():
    """
    모든 세션 목록 조회

    Returns:
        List[Session]: 세션 목록
    """
    return list(sessions_store.values())


@router.get("/{session_id}", response_model=Session)
async def get_session(session_id: str):
    """
    특정 세션 조회

    Args:
        session_id: 세션 ID

    Returns:
        Session: 세션 정보

    Raises:
        HTTPException: 세션을 찾을 수 없을 때
    """
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")

    return sessions_store[session_id]


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """
    세션 삭제

    Args:
        session_id: 세션 ID

    Returns:
        dict: 삭제 성공 메시지

    Raises:
        HTTPException: 세션을 찾을 수 없을 때
    """
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions_store[session_id]
    return {"message": "Session deleted successfully"}
