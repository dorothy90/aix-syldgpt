from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class Artifact(BaseModel):
    type: str  # "html" | "image" (image는 추후)
    mime: str  # "text/html" | "image/png" ...
    data: str  # html string or base64
    title: Optional[str] = None


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    artifacts: Optional[List[Artifact]] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    message: str


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
