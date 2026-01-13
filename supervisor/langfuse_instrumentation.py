"""Langfuse 계측 유틸 (Supervisor 실습용)

목표:
- `csv_supervisor.ipynb` / `supervisor/main.py`에서 `invoke()` 호출 시
  Langfuse Trace/Span/Generation이 생성되도록 LangChain 콜백을 주입합니다.
"""

from __future__ import annotations

import os
from datetime import datetime
import re
import json
import inspect
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv("/Users/daehwankim/Documents/langgraph-tutorial-main/.env", override=True)

_DEFAULT_LANGFUSE_HOST = "http://localhost:3001"  # docker-compose.yml: 3001:3000
_handler = None
_session_id = None

_OPER_RE = re.compile(r"(OP\\d{3})")
_KNOWN_PARAS = {
    "TEMP",
    "PRESSURE",
    "HUMIDITY",
    "FLOW",
    "VOLTAGE",
    "CURRENT",
    "VIBRATION",
    "SPEED",
}

# #region agent log
_DEBUG_LOG_PATH = "/Users/daehwankim/Documents/langgraph-tutorial-main/.cursor/debug.log"


def _dbg(
    hypothesisId: str, location: str, message: str, data: Dict[str, Any], *, runId: str
):
    # FORBIDDEN: logging secrets (API keys etc.)
    payload = {
        "sessionId": "debug-session",
        "runId": runId,
        "hypothesisId": hypothesisId,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": __import__("time").time(),
    }
    try:
        open(_DEBUG_LOG_PATH, "a").write(json.dumps(payload) + "\n")
    except Exception:
        pass


# #endregion


def _get_session_id() -> str:
    global _session_id
    if _session_id is not None:
        return _session_id
    _session_id = os.getenv("LANGFUSE_SESSION_ID") or (
        "csv-supervisor-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return _session_id


def get_langfuse_handler():
    """환경변수가 준비돼 있으면 Langfuse CallbackHandler를 반환합니다.

    필요한 환경변수:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - (선택) LANGFUSE_HOST (기본값: http://localhost:3001)
    """

    global _handler
    if _handler is not None:
        return _handler

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", _DEFAULT_LANGFUSE_HOST)

    _dbg(
        "A",
        "langfuse_instrumentation.py:get_langfuse_handler:env",
        "Loaded env flags",
        {
            "has_public_key": bool(public_key),
            "has_secret_key": bool(secret_key),
            "host": host,
        },
        runId="post-fix",
    )

    if not public_key or not secret_key:
        return None

    try:
        # 공식 문서 기준: langfuse.langchain.CallbackHandler
        from langfuse.langchain import CallbackHandler  # type: ignore
    except Exception as e:  # pragma: no cover
        # 노트북에서 “패키지 설치 누락”을 바로 알 수 있게 출력
        _dbg(
            "B",
            "langfuse_instrumentation.py:get_langfuse_handler:import",
            "Failed to import langfuse.langchain.CallbackHandler",
            {"error_type": type(e).__name__, "error": str(e)},
            runId="post-fix",
        )
        print(f"⚠️ Langfuse 패키지를 import 할 수 없습니다: {e}")
        print("   해결: `uv add langfuse` 후 `uv sync` 또는 `pip install langfuse`")
        return None

    environment = os.getenv("LANGFUSE_ENVIRONMENT", "local")
    release = os.getenv("LANGFUSE_RELEASE", "csv-supervisor")

    # 런타임 증거: 현재 설치된 CallbackHandler 시그니처/소스 확인
    try:
        _dbg(
            "A",
            "langfuse_instrumentation.py:get_langfuse_handler:handler_sig",
            "Resolved CallbackHandler",
            {
                "signature": str(inspect.signature(CallbackHandler)),
                "sourcefile": inspect.getsourcefile(CallbackHandler),
            },
            runId="post-fix",
        )
    except Exception as e:
        _dbg(
            "A",
            "langfuse_instrumentation.py:get_langfuse_handler:handler_sig",
            "Failed to inspect CallbackHandler",
            {"error_type": type(e).__name__, "error": str(e)},
            runId="post-fix",
        )

    # 핵심 호환 처리:
    # - 현재 .venv의 CallbackHandler는 secret_key/base_url을 받지 않음
    # - 대신 Langfuse client를 먼저 초기화하고, handler에는 public_key만 전달
    try:
        from langfuse import Langfuse  # type: ignore

        _ = Langfuse(public_key=public_key, secret_key=secret_key, base_url=host)
        _dbg(
            "C",
            "langfuse_instrumentation.py:get_langfuse_handler:client_init",
            "Initialized Langfuse client",
            {"client_created": True},
            runId="post-fix",
        )
    except Exception as e:
        _dbg(
            "C",
            "langfuse_instrumentation.py:get_langfuse_handler:client_init",
            "Failed to initialize Langfuse client",
            {"error_type": type(e).__name__, "error": str(e)},
            runId="post-fix",
        )

    # NOTE: 현재 설치된 Langfuse LangChain handler는 `set_trace_params()`가 없습니다.
    # trace/session/user/tags는 invoke 시 config.metadata의 langfuse_* 키로 전달합니다.
    _handler = CallbackHandler(public_key=public_key, update_trace=True)

    _dbg(
        "E",
        "langfuse_instrumentation.py:get_langfuse_handler:ready",
        "Langfuse handler created",
        {"update_trace": True},
        runId="post-fix",
    )

    print(f"✅ Langfuse enabled: host={host}, session_id={_get_session_id()}")
    return _handler


def invoke_with_langfuse(
    runnable: Any,
    query: str,
    *,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_name: str = "csv_supervisor_invoke",
):
    """LangGraph/LangChain Runnable을 Langfuse callbacks로 invoke 합니다.

    - runnable: `.invoke(input, config=...)`를 지원하는 객체 (예: compiled graph)
    - query: 사용자 질문
    """

    handler = get_langfuse_handler()
    input_payload = {"messages": [{"role": "user", "content": query}]}

    if handler is None:
        return runnable.invoke(input_payload)

    # --- query에서 최소 메타데이터 추출(검색/필터 편의) ---
    oper = None
    m = _OPER_RE.search(query)
    if m:
        oper = m.group(1)

    para = None
    for p in _KNOWN_PARAS:
        if p in query:
            para = p
            break

    chart_type = None
    if any(k in query.lower() for k in ["bar"]) or any(
        k in query for k in ["막대", "바"]
    ):
        chart_type = "bar"
    if any(k in query.lower() for k in ["line"]) or "선 그래프" in query:
        chart_type = chart_type or "line"

    intent = "filter"
    if any(k in query for k in ["연관", "관련", "related"]):
        intent = "relation"
    if any(k in query for k in ["그래프", "차트", "시각화", "plot", "trend", "시계열"]):
        intent = "viz" if intent == "filter" else f"{intent}+viz"

    environment = os.getenv("LANGFUSE_ENVIRONMENT", "local")
    release = os.getenv("LANGFUSE_RELEASE", "csv-supervisor")

    meta: Dict[str, Any] = {
        "langfuse_session_id": _get_session_id(),
        "langfuse_user_id": os.getenv("LANGFUSE_USER_ID", "local"),
        "langfuse_tags": ["csv", "supervisor", environment, release, intent]
        + (tags or []),
        "query": query,
        "intent": intent,
    }
    if oper:
        meta["oper"] = oper
    if para:
        meta["para"] = para
    if chart_type:
        meta["chart_type"] = chart_type

    if metadata:
        meta.update(metadata)

    return runnable.invoke(
        input_payload,
        config={
            "callbacks": [handler],
            "metadata": meta,
            "run_name": run_name,
        },
    )
