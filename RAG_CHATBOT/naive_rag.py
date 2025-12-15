# %%
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# macOS에서 faiss/numpy 등의 OpenMP 런타임 중복 로딩으로 크래시가 날 수 있어
# 임시 우회(필요 시 추후 제거/정리)
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from func.docs import MongoEmbeddingRetrievalChain

# API 키 정보 로드
load_dotenv(override=True)
model_name = os.getenv("RETRIEVE_CHAIN_MODEL")
base_url = os.getenv("OPENROUTER_BASE_URL")
api_key = os.getenv("OPENROUTER_API_KEY")
# mongodb 로드합니다.
mongodb = MongoEmbeddingRetrievalChain().create_chain()

# retriever와 chain을 생성합니다.
retriever = mongodb.retriever
chain = mongodb.chain

# %%
# Test
if __name__ == "__main__":
    from langchain_teddynote.messages import messages_to_history

    query = "workd2Vec이 뭐야"
    seacrh_result_retriever = retriever.invoke(query)
    seacrh_result_chain = chain.invoke(
        {
            "question": query,
            "context": seacrh_result_retriever,
            "chat_history": messages_to_history([]),  # 빈 대화 기록 추가
        }
    )

    print(f"retriever result: {seacrh_result_retriever}")
    print(f"chain result: {seacrh_result_chain}")

# %%

## 랭그래프로 naive rag 구현
# State 정의
from typing import Annotated, TypedDict, Any, Dict, List, Optional
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    question: Annotated[str, "Question"]
    context: Annotated[str, "Context"]
    answer: Annotated[str, "Answer"]
    messages: Annotated[list, add_messages]
    artifacts: Annotated[list, "Artifacts"]
    route: Annotated[str, "Route"]
    # mes 관련 중간 상태는 ReAct agent 내부에서 처리


# node 정의
from langchain_teddynote.messages import messages_to_history
from func.utils import format_searched_docs


# 문서 검색 노드
def retrieve_document(state: GraphState) -> GraphState:
    # 질문을 상태에서 가져옴
    latest_question = state["question"]
    # 문서에서 검색하여 관련있는 문서 가져오기
    retrieve_docs = retriever.invoke(latest_question)
    # 검색된 문서 형식화 (프롬프트에 넣을 때 더 정형화해서 넣기)
    retrieved_docs = format_searched_docs(retrieve_docs)
    # 검색된 문서를 state의 context 키에 저장
    return {"context": retrieved_docs}


# 답변 생성 노드
def llm_answer(state: GraphState) -> GraphState:
    # 질문을 상태에서 가져옴
    latest_question = state["question"]
    # 검색된 문서를 상태에서 가져옴
    context = state.get("context", "")

    messages = state.get("messages", [])
    # 체인을 스트리밍으로 호출하여 답변 생성
    # stream()을 사용하면 LangGraph의 stream_mode="messages"가 작동함
    response = ""
    for chunk in chain.stream(
        {
            "question": latest_question,
            "context": context,
            "chat_history": messages_to_history(messages) if messages else "",
        }
    ):
        response += chunk

    return {
        "answer": response,
        "messages": [
            HumanMessage(content=latest_question),
            AIMessage(content=response),
        ],
    }


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Query Rewrite 프롬프트 정의
from langchain_core.prompts import load_prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import json
import re
import urllib.request
import urllib.error

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
re_write_prompt = load_prompt(str(_PROMPTS_DIR / "rewrite.yaml"))

question_rewriter = (
    re_write_prompt
    | ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key, temperature=0)
    | StrOutputParser()
)


# Query Rewrite 노드
def query_rewrite(state: GraphState) -> GraphState:
    latest_question = state["question"]
    question_rewritten = question_rewriter.invoke({"question": latest_question})
    return {"question": str(question_rewritten).strip()}


# Route (query_rewrite 이후 mes vs retrieve vs direct)
route_prompt = ChatPromptTemplate.from_template(
    """너는 라우터다. 아래 질문을 보고 다음 중 하나만 결정해라.

- mes: LOT 상태/이력처럼 MES API 호출로 HTML을 만들어야 하는 질문
- retrieve: 사내 문서/정의/절차 등 검색(RAG)이 필요한 질문
- direct: 일반 상식/간단한 대화로 바로 답할 수 있는 질문

출력은 반드시 한 단어로만: "mes" 또는 "retrieve" 또는 "direct"

질문: {question}
"""
)

route_decider = (
    route_prompt
    | ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key, temperature=0)
    | StrOutputParser()
)


def route_main(state: GraphState) -> str:
    q = state["question"]
    decision = route_decider.invoke({"question": q}).strip().lower()

    # 모델 출력이 약간 흔들려도 안전하게 처리
    if "mes" in decision:
        return "mes"
    if "direct" in decision:
        return "direct"
    if "retrieve" in decision:
        return "retrieve"
    return "retrieve"


def _http_get_json(url: str, timeout_sec: float = 5.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        return {"error": f"HTTPError {e.code}", "detail": body}
    except Exception as e:
        return {"error": type(e).__name__, "detail": str(e)}


@tool
def mes_get_lot_status(lot_id: str) -> Dict[str, Any]:
    """MES API로 LOT 현재 상태를 조회합니다. lot_id는 7자리(3자리 lotcode + 4자리 숫자)입니다."""
    lot_id = str(lot_id).strip()
    return _http_get_json(f"http://127.0.0.1:8000/api/mes/lot_status/{lot_id}")


@tool
def mes_get_lot_history(lot_id: str) -> Dict[str, Any]:
    """MES API로 LOT 이력을 조회합니다. lot_id는 7자리(3자리 lotcode + 4자리 숫자)입니다."""
    lot_id = str(lot_id).strip()
    return _http_get_json(f"http://127.0.0.1:8000/api/mes/lot_history/{lot_id}")


_MES_TOOLS = [mes_get_lot_status, mes_get_lot_history]

_mes_model = ChatOpenAI(
    model=model_name,
    base_url=base_url,
    api_key=api_key,
    temperature=0,
)

# REACT_AGENT/04-React-Agent.ipynb 스타일: prebuilt ReAct agent
_mes_agent_executor = create_react_agent(_mes_model, _MES_TOOLS)


def _html_escape(s: Any) -> str:
    txt = "" if s is None else str(s)
    return (
        txt.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _render_mes_html(kind: str, lot_id: str, payload: Dict[str, Any]) -> str:
    data = payload.get("data") or {}
    found = bool(data.get("found"))

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
        body = f"<div class='mes-empty'>{_html_escape(data.get('error') or '조회 결과가 없습니다.')}</div>"
        return style + header + sub + body + footer

    if kind == "lot_history":
        events = data.get("events") or []
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
    row = (data.get("row") or {}) if isinstance(data, dict) else {}
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


def mes_agent(state: GraphState) -> GraphState:
    """
    단일 ReAct Agent:
    - LLM이 tool을 선택/호출(mes_get_lot_status|mes_get_lot_history)
    - tool 결과로 고정 HTML 템플릿 생성
    - answer + artifacts 반환
    """
    q = state["question"]

    system = SystemMessage(
        content=(
            "너는 MES 에이전트다.\n"
            "- LOT 상태/이력 조회가 필요하면 반드시 제공된 tool을 호출해라.\n"
            "- lot_id는 7자리(3자리 lotcode + 4자리 숫자) 예: ABC0001\n"
            "- 최종 응답은 한글로 짧게 요약만 하고(HTML은 시스템이 렌더링), 불필요한 장황한 설명은 하지 마라.\n"
        )
    )

    final = _mes_agent_executor.invoke({"messages": [system, HumanMessage(content=q)]})
    msgs = final.get("messages", [])

    # 가장 최근 tool 결과/호출을 찾아 kind/lot_id/payload 확보
    kind = None
    lot_id = None
    payload = None
    for m in reversed(msgs):
        if isinstance(m, ToolMessage) and m.name in (
            "mes_get_lot_status",
            "mes_get_lot_history",
        ):
            kind = "lot_status" if m.name == "mes_get_lot_status" else "lot_history"
            try:
                payload = json.loads(m.content)
            except Exception:
                payload = {"data": {"found": False, "error": "tool 결과 파싱 실패"}}
            # lot_id는 payload에 포함돼 있으면 사용
            if isinstance(payload, dict):
                lot_id = payload.get("lot_id") or lot_id
            break

    if not kind or not payload:
        answer = "LOT 번호(예: ABC0001)를 포함해서 현재 상태 또는 이력을 물어봐 주세요."
        return {
            "answer": answer,
            "messages": [HumanMessage(content=q), AIMessage(content=answer)],
        }

    if not lot_id:
        # tool이 호출됐는데 lot_id가 없다면 payload에서 복구
        lot_id = (payload.get("lot_id") if isinstance(payload, dict) else None) or ""

    data = payload.get("data") if isinstance(payload, dict) else {}
    found = bool((data or {}).get("found"))

    if not lot_id:
        answer = "LOT 번호(예: ABC0001)를 포함해서 다시 질문해 주세요."
    elif not found:
        answer = f"{lot_id} 조회 결과가 없습니다."
    else:
        answer = f"{lot_id} {('이력' if kind == 'lot_history' else '현재 상태')}입니다."

    html = _render_mes_html(kind=kind, lot_id=lot_id, payload=payload)
    artifacts = [{"type": "html", "mime": "text/html", "data": html, "title": kind}]

    return {
        "answer": answer,
        "artifacts": artifacts,
        "messages": [HumanMessage(content=q), AIMessage(content=answer)],
    }


# %%


# langgraph workflow 초기화
workflow = StateGraph(GraphState)

# workflow 노드 추가
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("query_rewrite", query_rewrite)
workflow.add_node("llm_answer", llm_answer)
workflow.add_node("mes_agent", mes_agent)

# workflow 엣지 추가
workflow.add_conditional_edges(
    "query_rewrite",
    route_main,
    {
        "mes": "mes_agent",
        "retrieve": "retrieve",
        "direct": "llm_answer",
    },
)
workflow.add_edge("retrieve", "llm_answer")
workflow.add_edge("llm_answer", END)
workflow.add_edge("mes_agent", END)

# workflow 진입점 설정
workflow.set_entry_point("query_rewrite")

# 체크포인터 설정
memory = MemorySaver()

# 컴파일
app = workflow.compile(checkpointer=memory)
# %%
# 그래프 시각화 및 샘플 실행 (직접 실행 시에만 동작)
if __name__ == "__main__":
    from langchain_teddynote.graphs import visualize_graph

    visualize_graph(app)

    from langchain_core.runnables import RunnableConfig
    from langchain_teddynote.messages import invoke_graph, stream_graph, random_uuid

    # config 설정 (재귀 리밋, thread_id)
    config = RunnableConfig(
        recursion_limit=5, configurable={"thread_id": random_uuid()}
    )

    # 질문 입력
    inputs: GraphState = {"question": "attentiom 메커니즘에 대해서 알려줘"}

    invoke_graph(app, inputs, config)


# %%
