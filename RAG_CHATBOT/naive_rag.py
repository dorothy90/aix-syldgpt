# %%
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

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
    mes_ctx: Annotated[Dict[str, Any], "MESContext"]


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
from langchain_core.messages import HumanMessage, AIMessage
import re

from func.mes_agent import mes_agent

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
re_write_prompt = load_prompt(str(_PROMPTS_DIR / "rewrite.yaml"))

question_rewriter = (
    re_write_prompt
    | ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key, temperature=0)
    | StrOutputParser()
)


# Query Rewrite 노드
def query_rewrite(state: GraphState) -> GraphState:
    q = str(state.get("question") or "").strip()
    mes_ctx = state.get("mes_ctx") or {}

    # MES 모드 종료/초기화 키워드(사용자가 일반 질문으로 돌아가고 싶을 때)
    if re.search(
        r"(mes\s*종료|메스\s*종료|모드\s*종료|일반\s*질문|리셋|초기화)", q, re.I
    ):
        return {"question": q, "mes_ctx": {}}

    # MES 모드(active)에서는 리라이트를 건너뛰어 식별자/필터 토큰을 보존
    if mes_ctx.get("active"):
        return {"question": q}

    # LOT 식별자는 리라이트로 변형될 여지가 있어 그대로 보존
    if re.search(r"\b[A-Z]{3}\d{4}\b", q):
        return {"question": q}

    question_rewritten = question_rewriter.invoke({"question": q})
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
    mes_ctx = state.get("mes_ctx") or {}

    # 한 번 MES로 진입하면 이후 질의는 전부 MES agent로 처리(재질문/필터 누적은 mes_agent가 담당)
    if mes_ctx.get("active"):
        return "mes"

    # LOT 패턴은 결정론적으로 MES로 라우팅 (초기 라우팅 안정화)
    if re.search(r"\b[A-Z]{3}\d{4}\b", q or ""):
        return "mes"

    decision = route_decider.invoke({"question": q}).strip().lower()

    # 모델 출력이 약간 흔들려도 안전하게 처리
    if "mes" in decision:
        return "mes"
    if "direct" in decision:
        return "direct"
    if "retrieve" in decision:
        return "retrieve"
    return "retrieve"


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
