from __future__ import annotations

from dotenv import load_dotenv
import os
import json
import re
import urllib.request
import urllib.error
import io
import base64
from typing import Any, Dict, List, Optional, Union, Literal
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

from rich import print

# Option B: 이 모듈이 독립적으로 .env를 로드하고 모델 설정을 읽는다.
load_dotenv(override=True)
_MODEL_NAME = os.getenv("RETRIEVE_CHAIN_MODEL")
_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
_API_KEY = os.getenv("OPENROUTER_API_KEY")


def _http_post_json(url: str, body: Dict[str, Any], timeout_sec: float = 5.0) -> Any:
    """MES 백엔드 API 호출 유틸리티"""
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
        if isinstance(parsed, list):
            return parsed
        return [{"error": f"HTTPError {e.code}", "detail": raw}]
    except Exception as e:
        return [{"error": type(e).__name__, "detail": str(e)}]


# -------------------- MES Tools --------------------
@tool
def mes_get_lot_status(lot_id: str) -> Any:
    """
    MES API로 LOT 현재 상태를 조회합니다.

    Args:
        lot_id: LOT 번호 (예: ABC0001). 7자리(3자리 lotcode + 4자리 숫자) 형태입니다.

    Returns:
        LOT의 현재 상태 정보 (product, step, eqp, status, updated_at 포함)
    """
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
    """
    MES API로 LOT 이력을 조회합니다.

    Args:
        lot_id: LOT 번호 (예: ABC0001)
        filters: 필터 조건 리스트. 각 필터는 {"field": str, "op": str, "value": Any} 형태.
                 - field: event_time, event, step, eqp, details 중 하나
                 - op: eq, contains, in, gte, lte 중 하나
        limit: 최대 조회 개수

    Returns:
        LOT 이력 리스트 (event_time, event, step, eqp, details 포함)
    """
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


# WIP Report CSV 파일 경로 설정
_WIP_CSV_PATH = (
    Path(__file__).parent.parent.parent / "backend" / "app" / "data" / "wip_report.csv"
)


@tool
def mes_get_wip_report(
    step: Optional[str] = None,
    chamb_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    WIP Report를 조회하여 날짜별 lot count를 bar chart로 시각화합니다.

    Args:
        step: 필터링할 step (예: "step001"). None이면 전체 step 대상으로 집계합니다.
        chamb_id: 컬러링할 chamber ID (예: "CH01").
                  - 지정 시: 해당 chamb_id만 표시 (단일 색상)
                  - 미지정 시: 전체 chamb_id별로 컬러링 (stacked bar chart)

    Returns:
        날짜별 lot count 데이터 및 bar chart 이미지(base64)
        - x축: end_tm의 날짜 (yyyy-mm-dd)
        - y축: lot count
        - chamb_id별 컬러링
    """

    # 한글 폰트 설정 (matplotlib 백엔드)
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = [
        "AppleGothic",
        "Malgun Gothic",
        "NanumGothic",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    try:
        df = pd.read_csv(_WIP_CSV_PATH)
    except FileNotFoundError:
        return [
            {
                "error": "FileNotFoundError",
                "detail": f"WIP CSV 파일을 찾을 수 없습니다: {_WIP_CSV_PATH}",
            }
        ]
    except Exception as e:
        return [{"error": type(e).__name__, "detail": str(e)}]

    # end_tm을 datetime으로 파싱하고 날짜만 추출
    df["end_tm"] = pd.to_datetime(df["end_tm"])
    df["date"] = df["end_tm"].dt.strftime("%Y-%m-%d")

    # step 필터링 적용
    if step:
        df = df[df["step"].str.contains(step, case=False, na=False)]
        print(f"[DEBUG]   - step 필터링 후: {len(df)} rows")

    # chamb_id 필터링 (지정된 경우에만)
    if chamb_id:
        df = df[df["chamb_id"].str.contains(chamb_id, case=False, na=False)]

    if df.empty:
        return [{"error": "NoData", "detail": "조건에 맞는 WIP 데이터가 없습니다."}]

    # Bar chart 생성
    fig, ax = plt.subplots(figsize=(14, 6))

    if chamb_id:
        # chamb_id가 지정된 경우: 해당 chamb만 단일 색상으로 표시
        date_counts = df.groupby("date")["lot_id"].count().sort_index()
        colors = plt.cm.tab20.colors
        bars = ax.bar(
            date_counts.index, date_counts.values, color=colors[0], edgecolor="white"
        )
        ax.set_ylabel("LOT Count")
        ax.set_xlabel("Date")
        title_parts = ["WIP Report - 날짜별 LOT Count"]
        if step:
            title_parts.append(f"Step: {step}")
        title_parts.append(f"Chamber: {chamb_id}")
        ax.set_title(" | ".join(title_parts))
    else:
        # chamb_id가 없는 경우: chamb별 컬러링 (stacked bar chart)
        pivot = df.pivot_table(
            index="date",
            columns="chamb_id",
            values="lot_id",
            aggfunc="count",
            fill_value=0,
        )
        pivot = pivot.sort_index()

        # Stacked bar chart
        colors = plt.cm.tab20.colors
        bottom = None
        for idx, col in enumerate(pivot.columns):
            color = colors[idx % len(colors)]
            if bottom is None:
                ax.bar(
                    pivot.index, pivot[col], label=col, color=color, edgecolor="white"
                )
                bottom = pivot[col].values
            else:
                ax.bar(
                    pivot.index,
                    pivot[col],
                    bottom=bottom,
                    label=col,
                    color=color,
                    edgecolor="white",
                )
                bottom = bottom + pivot[col].values

        ax.legend(
            title="Chamber", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8
        )
        ax.set_ylabel("LOT Count")
        ax.set_xlabel("Date")
        title_suffix = f" (Step: {step})" if step else ""
        ax.set_title(f"WIP Report - 날짜별 LOT Count{title_suffix}")

    # X축 라벨 회전
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # 이미지를 base64로 인코딩
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    # 집계 데이터 준비 (날짜별)
    date_counts = df.groupby("date")["lot_id"].count().to_dict()
    total_lots = df["lot_id"].count()

    return [
        {
            "chart_image": img_b64,
            "date_counts": date_counts,
            "total_lots": int(total_lots),
            "filter_step": step,
            "filter_chamb": chamb_id,
        }
    ]


# -------------------- MES Agent (create_agent 패턴) --------------------

# MES Agent용 시스템 프롬프트
MES_SYSTEM_PROMPT = """당신은 MES(Manufacturing Execution System) 전문 어시스턴트입니다.

사용자가 LOT 상태나 이력에 대해 질문하면 적절한 도구를 사용하여 정보를 조회하고 답변합니다.

## 사용 가능한 도구:
1. **mes_get_lot_status**: LOT의 현재 상태/정보(product, step, eqp, status, updated_at)를 조회
2. **mes_get_lot_history**: LOT의 이력(event_time, event, step, eqp, details)을 조회
3. **mes_get_wip_report**: WIP 데이터 조회 및 날짜별 lot count bar chart 시각화
   - x축: end_tm의 날짜 (yyyy-mm-dd)
   - y축: lot count
   - step: 필터링할 step (예: "step001", 옵션) - 데이터 필터링에 사용
   - chamb_id: 컬러링할 chamber (예: "CH01", 옵션)
     - 지정 시: 해당 chamber만 표시 (단일 색상)
     - 미지정 시: 전체 chamber별로 컬러링 (stacked bar chart)

## 응답 규칙:
- LOT 번호는 반드시 7자리 형태(예: ABC0001)로 사용합니다.
- 사용자가 LOT 번호를 언급하지 않으면, LOT 번호를 먼저 요청하세요.
- 이력 조회 시 날짜 범위가 필요하면 filters를 활용하세요.
- WIP 조회 시 step으로 데이터를 필터링하고, chamb_id로 컬러링을 구분할 수 있습니다.
- 조회 결과가 없으면 명확하게 안내합니다.
- 응답은 한국어로 친절하게 제공합니다.

## 필터 사용 예시:
- 특정 날짜 이후: {"field": "event_time", "op": "gte", "value": "2025-01-01 00:00:00"}
- 특정 이벤트만: {"field": "event", "op": "eq", "value": "TRACK_IN"}
- 특정 step만: {"field": "step", "op": "contains", "value": "DIFF"}

## WIP 조회 예시:
- 전체 WIP 현황 (날짜별, 전체 chamber 컬러링): mes_get_wip_report()
- 특정 step의 WIP (날짜별, 전체 chamber 컬러링): mes_get_wip_report(step="step001")
- 특정 chamber의 WIP (날짜별, 단일 색상): mes_get_wip_report(chamb_id="CH01")
- step 필터 + chamber 컬러: mes_get_wip_report(step="step001", chamb_id="CH01")

## 중요: 응답 형식
- 도구 호출 결과는 별도의 HTML 카드로 자동 표시됩니다.
- 따라서 **테이블이나 표를 직접 만들지 마세요**.
- 조회 결과에 대해 간단한 요약/설명만 1-2문장으로 작성하세요.
- 예시: "ABC0001 LOT은 현재 DIFF-10 공정에서 RUN 상태입니다."
- WIP 예시: "현재 전체 WIP는 1,234개 LOT이며, 12월 20일에 가장 많은 LOT이 처리되었습니다."
"""

# MES 전용 도구 리스트
MES_TOOLS = [mes_get_lot_status, mes_get_lot_history, mes_get_wip_report]

# MES Agent용 모델
_mes_model = ChatOpenAI(
    model=_MODEL_NAME,
    base_url=_BASE_URL,
    api_key=_API_KEY,
    temperature=0,
)

# create_agent로 MES Agent 생성
mes_react_agent = create_agent(
    model=_mes_model,
    tools=MES_TOOLS,
    system_prompt=MES_SYSTEM_PROMPT,
)


# -------------------- HTML 렌더링 (기존 유지) --------------------
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

    if kind == "wip_report":
        # WIP Report: bar chart 이미지 렌더링
        img_b64 = first.get("chart_image", "")
        filter_step = first.get("filter_step") or "전체"
        filter_chamb = first.get("filter_chamb") or "전체"
        total_lots = first.get("total_lots", 0)

        wip_header = (
            f"<div class='mes-card'><div class='mes-title'>MES: WIP Report</div>"
        )
        wip_sub = (
            f"<div class='mes-sub'>"
            f"Step: <span class='mes-badge'>{_html_escape(filter_step)}</span> | "
            f"Chamber: <span class='mes-badge'>{_html_escape(filter_chamb)}</span> | "
            f"Total LOTs: <span class='mes-badge'>{total_lots}</span>"
            f"</div>"
        )
        img_html = (
            f"<div style='text-align:center;'>"
            f"<img src='data:image/png;base64,{img_b64}' style='max-width:100%;border-radius:8px;'/>"
            f"</div>"
        )
        return style + wip_header + wip_sub + img_html + footer

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


# -------------------- LangGraph 노드용 래퍼 함수 --------------------
def mes_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    naive_rag.py의 LangGraph 노드로 연결되는 MES 처리 함수.
    내부적으로 create_agent로 만든 mes_react_agent를 호출합니다.
    """
    print(f"\n{'='*60}")

    q = state["question"]
    mes_ctx = state.get("mes_ctx") or {}
    print(f"question: {q}")
    print(f"mes_ctx: {mes_ctx}")

    # 이전 대화 히스토리 구성
    messages = state.get("messages", [])
    # mes_react_agent 호출
    result = mes_react_agent.invoke({"messages": messages + [HumanMessage(content=q)]})

    print(f"result: {result}")

    # 결과에서 마지막 AI 메시지 추출
    ai_messages = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]

    answer = ai_messages[-1].content if ai_messages else "MES 조회에 실패했습니다."

    # LOT ID 추출 (응답에서 LOT 패턴 찾기)
    lot_match = re.search(r"\b([A-Z]{3}\d{4})\b", q)
    lot_id = lot_match.group(1) if lot_match else mes_ctx.get("last_lot_id", "")

    # 조회 종류 판별 및 도구 호출 결과에서 payload 추출
    kind = "lot_history"
    payload = None

    # ToolMessage에서 도구 실행 결과 추출
    from langchain_core.messages import ToolMessage

    for idx, msg in enumerate(result.get("messages", [])):

        # AIMessage인 경우 tool_calls 상세 분석
        if isinstance(msg, AIMessage):

            # tool_calls가 비어있지만 additional_kwargs에 있는 경우 확인
            ak_tool_calls = msg.additional_kwargs.get("tool_calls", [])
            if not msg.tool_calls and ak_tool_calls:
                print(
                    f"!!! WARNING: tool_calls가 비어있지만 additional_kwargs에 있음 !!!"
                )
                print(f"!!! additional_kwargs['tool_calls']: {ak_tool_calls}")

        if isinstance(msg, ToolMessage):
            try:
                # 도구 결과 파싱
                tool_result = msg.content
                if isinstance(tool_result, str):
                    print(
                        f"[DEBUG]   content (str): {tool_result[:200]}..."
                        if len(tool_result) > 200
                        else f"[DEBUG]   content (str): {tool_result}"
                    )
                    tool_result = json.loads(tool_result)
                payload = tool_result

                # 도구 이름으로 kind 판별
                if msg.name == "mes_get_lot_status":
                    kind = "lot_status"
                elif msg.name == "mes_get_lot_history":
                    kind = "lot_history"
                elif msg.name == "mes_get_wip_report":
                    kind = "wip_report"
            except (json.JSONDecodeError, TypeError) as e:
                payload = None

    # HTML 아티팩트 생성
    artifacts = []

    if payload and (lot_id or kind == "wip_report"):
        print(f"[DEBUG]   - 조건 충족! HTML 렌더링 시작")
        html = _render_mes_html(kind=kind, lot_id=lot_id, payload=payload)
        artifacts = [{"type": "html", "mime": "text/html", "data": html, "title": kind}]
        print(f"[DEBUG]   - artifacts 생성 완료 (HTML 길이: {len(html)} chars)")
    else:
        print(f"[DEBUG]   - 조건 미충족! artifacts 비어있음")

    result_dict = {
        "answer": answer,
        "artifacts": artifacts,
        "messages": [HumanMessage(content=q), AIMessage(content=answer)],
        "mes_ctx": {
            "active": True,
            "last_kind": kind,
            "last_lot_id": lot_id,
        },
    }
    return result_dict


# -------------------- 직접 실행 테스트 --------------------
if __name__ == "__main__":
    from langchain_teddynote.messages import stream_graph

    # 테스트: create_agent로 만든 MES Agent 실행
    print("=== MES Agent 테스트 (create_agent 패턴) ===\n")

    stream_graph(
        mes_react_agent,
        inputs={"messages": [HumanMessage(content="ABC0001 LOT의 현재 상태를 알려줘")]},
    )
