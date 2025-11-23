from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class DocumentReference(BaseModel):
    """API에서 검색된 문서 참조 정보"""

    filename: str
    content: str
    url: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    message: str
    documents: Optional[List[DocumentReference]] = None  # API에서 검색된 문서들


class Session(BaseModel):
    session_id: str
    title: str
    messages: List[Message]
    created_at: str
    updated_at: Optional[str] = None


class SessionCreate(BaseModel):
    title: Optional[str] = None


class SessionUpdate(BaseModel):
    title: Optional[str] = None
