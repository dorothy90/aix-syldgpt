from __future__ import annotations

from dotenv import load_dotenv
import os
import json
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool


# Option B: 이 모듈이 독립적으로 .env를 로드하고 모델 설정을 읽는다.
load_dotenv(override=True)
_MODEL_NAME = os.getenv("RETRIEVE_CHAIN_MODEL")
_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
_API_KEY = os.getenv("OPENROUTER_API_KEY")


def _http_post_json(url: str, body: Dict[str, Any], timeout_sec: float = 5.0) -> Any:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else []
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8")
            parsed = json.loads(raw) if raw else None
        except Exception:
            parsed = None
        # MES 규칙: 항상 list[dict] 형태 유지
        if isinstance(parsed, list):
            return parsed
        return [{"error": f"HTTPError {e.code}", "detail": raw}]
    except Exception as e:
        return [{"error": type(e).__name__, "detail": str(e)}]


# -------------------- MES 필터 파서(LLM 구조화 출력) --------------------
_MES_ALLOWED_FIELDS = ["event_time", "event", "step", "eqp", "details"]
_MES_ALLOWED_OPS = ["eq", "contains", "in", "gte", "lte"]


class MESFilterCondition(BaseModel):
    field: Literal["event_time", "event", "step", "eqp", "details"]
    op: Literal["eq", "contains", "in", "gte", "lte"] = "eq"
    value: Union[str, int, float, List[str]]


class ParsedMESQuery(BaseModel):
    kind: Literal["lot_status", "lot_history"] = "lot_history"
    lot_id: Optional[str] = None
    filters: List[MESFilterCondition] = Field(default_factory=list)
    limit: Optional[int] = None
    # 날짜 범위처럼 연도 누락 시 추측하지 않고 확인 질문으로 유도
    needs_year: bool = False
    year_question: Optional[str] = None
    pending_time_range: Optional[Dict[str, str]] = (
        None  # {"start_mmdd":"11-01","end_mmdd":"11-20"}
    )


_mes_parser_llm = ChatOpenAI(
    model=_MODEL_NAME,
    base_url=_BASE_URL,
    api_key=_API_KEY,
    temperature=0,
).with_structured_output(ParsedMESQuery)


def parse_mes_query_with_llm(
    user_question: str,
    last_lot_id: Optional[str] = None,
    existing_filters: Optional[List[Dict[str, Any]]] = None,
) -> ParsedMESQuery:
    """
    자연어 질문 → (kind, lot_id, filters, limit) 구조화.
    - 누적형 재질문을 위해 last_lot_id/existing_filters를 힌트로 제공
    - 연도 없는 날짜 범위는 needs_year=true로 반환(추측 금지)
    """
    existing_filters = existing_filters or []
    system = SystemMessage(
        content=(
            "너는 MES 질의 파서다. 사용자의 자연어를 아래 스키마로 변환해라.\n"
            f"- kind는 lot_status 또는 lot_history\n"
            f"- field는 반드시 {_MES_ALLOWED_FIELDS} 중 하나\n"
            f"- op는 반드시 {_MES_ALLOWED_OPS} 중 하나\n"
            "- filters는 '데이터 포함/제외'를 결정하는 조건만 넣어라.\n"
            "- 사용자가 '그중에/그거에서/거기서/추가로/만'처럼 재질문하면, "
            "lot_id가 생략될 수 있으니 last_lot_id를 참고해도 된다.\n"
            "- 기존 조건(existing_filters)은 참고 정보일 뿐이며, 새로운 조건만 filters로 출력해라.\n"
            "- 날짜 범위에서 연도가 없으면 절대 추측하지 말고 needs_year=true로 하고 "
            "year_question에 '연도를 포함해서 다시 요청해주세요'는 한 문장을 넣어라.\n"
            "- 이 경우 event_time 필터는 만들지 말고, pending_time_range에 "
            '{"start_mmdd":"MM-DD","end_mmdd":"MM-DD"} 형태로만 저장해라.\n'
            "lot_id 자체를 찾기 위한 필터(예: details에 lot_id 포함)는 만들지 마라. lot_id는 lot_id 필드로만 지정해라.\n"
        )
    )
    hint = (
        f"last_lot_id={last_lot_id}\n"
        f"existing_filters={json.dumps(existing_filters, ensure_ascii=False)}"
    )
    human = HumanMessage(content=f"{hint}\n\nuser_question={user_question}")
    return _mes_parser_llm.invoke([system, human])


@tool
def mes_get_lot_status(lot_id: str) -> Any:
    """MES API로 LOT 현재 상태를 조회합니다. lot_id는 7자리(3자리 lotcode + 4자리 숫자)입니다."""
    lot_id = str(lot_id).strip()
    return _http_post_json(
        "http://127.0.0.1:8000/api/mes/lot_status",
        {"lot_id": lot_id},
    )


@tool
def mes_get_lot_history(
    lot_id: str,
    filters: Optional[List[Dict[str, Any]]] = None,
    limit: Optional[int] = None,
) -> Any:
    """MES API로 LOT 이력을 조회합니다. 필요하면 filters/limit를 함께 전달합니다."""
    lot_id = str(lot_id).strip()
    body: Dict[str, Any] = {"lot_id": lot_id}
    if filters:
        body["filters"] = filters
    if limit is not None:
        body["limit"] = limit
    return _http_post_json(
        "http://127.0.0.1:8000/api/mes/lot_history",
        body,
    )


def _html_escape(s: Any) -> str:
    txt = "" if s is None else str(s)
    return (
        txt.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _render_mes_html(kind: str, lot_id: str, payload: Any) -> str:
    rows: List[Dict[str, Any]] = payload if isinstance(payload, list) else []
    first: Dict[str, Any] = rows[0] if rows and isinstance(rows[0], dict) else {}
    is_error = bool(first.get("error"))
    found = bool(rows) and (not is_error)

    # 공통 wrapper(엔드포인트별로 항상 일정한 형식 유지)
    header = (
        f"<div class='mes-card'><div class='mes-title'>MES: {_html_escape(kind)}</div>"
    )
    footer = "</div>"

    style = """<style>
.mes-card{border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#fff}
.mes-title{font-weight:700;margin-bottom:8px}
.mes-sub{color:#6b7280;font-size:12px;margin-bottom:10px}
.mes-table{width:100%;border-collapse:collapse;font-size:13px}
.mes-table th,.mes-table td{border:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}
.mes-badge{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:12px}
.mes-empty{color:#6b7280}
</style>"""

    sub = f"<div class='mes-sub'>LOT: <span class='mes-badge'>{_html_escape(lot_id) if lot_id else '-'}</span></div>"

    if not found:
        if is_error:
            msg = first.get("detail") or first.get("error") or "오류가 발생했습니다."
        else:
            msg = "조회 결과가 없습니다."
        body = f"<div class='mes-empty'>{_html_escape(msg)}</div>"
        return style + header + sub + body + footer

    if kind == "lot_history":
        events = rows
        rows_html = ""
        for ev in events:
            rows_html += (
                "<tr>"
                f"<td>{_html_escape(ev.get('event_time'))}</td>"
                f"<td>{_html_escape(ev.get('event'))}</td>"
                f"<td>{_html_escape(ev.get('step'))}</td>"
                f"<td>{_html_escape(ev.get('eqp'))}</td>"
                f"<td>{_html_escape(ev.get('details'))}</td>"
                "</tr>"
            )
        table = (
            "<table class='mes-table'>"
            "<thead><tr><th>시간</th><th>이벤트</th><th>STEP</th><th>EQP</th><th>DETAILS</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )
        return style + header + sub + table + footer

    # lot_status
    row = first
    table = (
        "<table class='mes-table'>"
        "<tbody>"
        f"<tr><th>PRODUCT</th><td>{_html_escape(row.get('product'))}</td></tr>"
        f"<tr><th>STEP</th><td>{_html_escape(row.get('step'))}</td></tr>"
        f"<tr><th>EQP</th><td>{_html_escape(row.get('eqp'))}</td></tr>"
        f"<tr><th>STATUS</th><td>{_html_escape(row.get('status'))}</td></tr>"
        f"<tr><th>UPDATED_AT</th><td>{_html_escape(row.get('updated_at'))}</td></tr>"
        "</tbody></table>"
    )
    return style + header + sub + table + footer


def mes_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    naive_rag.py의 LangGraph 노드로 바로 연결되는 MES 처리 함수.
    - parse_mes_query_with_llm로 (kind, lot_id, filters, limit) 구조화
    - 백엔드 MES API 호출
    - 고정 HTML 템플릿 artifacts 반환
    """
    q = state["question"]
    mes_ctx = state.get("mes_ctx") or {}
    last_lot_id = mes_ctx.get("last_lot_id")
    existing_filters: List[Dict[str, Any]] = list(mes_ctx.get("filters") or [])
    last_limit = mes_ctx.get("limit")
    last_kind = mes_ctx.get("last_kind")

    parsed = parse_mes_query_with_llm(
        user_question=q, last_lot_id=last_lot_id, existing_filters=existing_filters
    )

    # 연도 누락 등으로 확인 질문이 필요한 경우(조회 보류)
    if getattr(parsed, "needs_year", False):
        pending = getattr(parsed, "pending_time_range", None)

        # pending_time_range는 반드시 MM-DD 형태여야 나중에 연도만 받아서 조립 가능
        def _is_mmdd(v: Any) -> bool:
            if not isinstance(v, str) or "-" not in v:
                return False
            a, b = v.split("-", 1)
            return (
                a.isdigit() and b.isdigit() and 1 <= int(a) <= 12 and 1 <= int(b) <= 31
            )

        if not (
            isinstance(pending, dict)
            and _is_mmdd(str(pending.get("start_mmdd", "")))
            and _is_mmdd(str(pending.get("end_mmdd", "")))
        ):
            pending = None
        answer = parsed.year_question or (
            "날짜 범위에 연도/월 정보가 필요합니다. 예: 2025년 12월 16일"
            if pending is None
            else "연도를 포함해서 다시 말해달라"
        )
        return {
            "answer": answer,
            "messages": [HumanMessage(content=q), AIMessage(content=answer)],
            "mes_ctx": {**mes_ctx, "pending_time_range": pending, "active": True},
        }

    kind = getattr(parsed, "kind", None) or last_kind or "lot_history"
    lot_id = (getattr(parsed, "lot_id", None) or last_lot_id or "").strip()

    if not lot_id:
        answer = "LOT 번호(예: ABC0001)를 포함해서 다시 질문해 주세요."
        return {
            "answer": answer,
            "messages": [HumanMessage(content=q), AIMessage(content=answer)],
            "mes_ctx": {**mes_ctx, "active": True},
        }

    # 연도만 뒤늦게 들어온 재질문 처리: pending_time_range + year로 event_time 범위를 생성해 누적
    pending = mes_ctx.get("pending_time_range")
    year_match = re.search(r"(19\d{2}|20\d{2})\s*년", q)
    injected_time_filters: List[Dict[str, Any]] = []
    if pending and year_match:
        y = int(year_match.group(1))
        try:
            start_mmdd = str(pending.get("start_mmdd") or "")
            end_mmdd = str(pending.get("end_mmdd") or "")
            sm, sd = [int(x) for x in start_mmdd.split("-")]
            em, ed = [int(x) for x in end_mmdd.split("-")]
            start = f"{y}-{sm:02d}-{sd:02d} 00:00:00"
            end = f"{y}-{em:02d}-{ed:02d} 23:59:59"
            injected_time_filters = [
                {"field": "event_time", "op": "gte", "value": start},
                {"field": "event_time", "op": "lte", "value": end},
            ]
            # 연도 확인이 끝났으니 pending 제거
            mes_ctx = {**mes_ctx, "pending_time_range": None}
        except Exception:
            injected_time_filters = []

    new_filters = [f.model_dump() for f in (getattr(parsed, "filters", None) or [])]
    if injected_time_filters:
        new_filters = injected_time_filters + new_filters
    # 최소 보정: event_time은 CSV에 시간까지 들어가므로 YYYY-MM-DD만 eq로 주면 매칭이 안 됨 → contains로 변경
    for f in new_filters:
        if (
            isinstance(f, dict)
            and (f.get("field") == "event_time")
            and (str(f.get("op") or "eq").lower() == "eq")
            and isinstance(f.get("value"), str)
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(f.get("value")).strip())
        ):
            f["op"] = "contains"
            f["value"] = str(f["value"]).strip()

    # 누적 정책: 항상 AND로 append
    combined_filters = existing_filters + new_filters if kind == "lot_history" else []

    # limit = getattr(parsed, "limit", None)
    # if limit is None:
    #     limit = last_limit
    # 변경 후: 사용자가 말한 경우(LLM이 명시적으로 limit을 낸 경우)에만 적용
    limit = getattr(parsed, "limit", None)
    # 실제 조회(결정론 필터링은 백엔드가 수행)
    if kind == "lot_status":
        payload = mes_get_lot_status.invoke({"lot_id": lot_id})
    else:
        tool_input: Dict[str, Any] = {"lot_id": lot_id}
        if combined_filters:
            tool_input["filters"] = combined_filters
        if limit is not None:
            tool_input["limit"] = limit
        payload = mes_get_lot_history.invoke(tool_input)

    rows = payload if isinstance(payload, list) else []
    first = rows[0] if rows and isinstance(rows[0], dict) else {}
    is_error = bool(first.get("error"))
    found = bool(rows) and (not is_error)

    if not lot_id:
        answer = "LOT 번호(예: ABC0001)를 포함해서 다시 질문해 주세요."
    elif not found:
        if is_error and (first.get("error") == "file_error"):
            answer = "MES 데이터 파일을 읽는 중 오류가 발생했습니다."
        else:
            answer = f"{lot_id} 조회 결과가 없습니다."
    else:
        answer = f"{lot_id} {('이력' if kind == 'lot_history' else '현재 상태')}입니다."

    html = _render_mes_html(kind=kind, lot_id=lot_id, payload=payload)
    artifacts = [{"type": "html", "mime": "text/html", "data": html, "title": kind}]

    return {
        "answer": answer,
        "artifacts": artifacts,
        "messages": [HumanMessage(content=q), AIMessage(content=answer)],
        "mes_ctx": {
            "active": True,
            "last_kind": kind,
            "last_lot_id": lot_id,
            "filters": combined_filters,
            "limit": limit,
            "pending_time_range": mes_ctx.get("pending_time_range"),
        },
    }
