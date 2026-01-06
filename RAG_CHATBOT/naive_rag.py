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
    # wads 관련 중간 상태
    wads_ctx: Annotated[Dict[str, Any], "WADSContext"]


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
        # 일반 답변 시 MES/WADS 모드 해제
        "mes_ctx": {},
        "wads_ctx": {},
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
from func.wads_agent import wads_agent

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

re_write_prompt = load_prompt(str(_PROMPTS_DIR / "rewrite.yaml"))

question_rewriter = (
    re_write_prompt
    | ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key, temperature=0)
    | StrOutputParser()
)


# Router Entry 노드 (passthrough - 바로 라우팅)
def router_entry(state: GraphState) -> GraphState:
    q = str(state.get("question") or "").strip()
    return {"question": q}


# Route (query_rewrite 이후 mes vs wads vs retrieve vs direct)
route_prompt = ChatPromptTemplate.from_template(
    """너는 라우터다. 아래 질문을 보고 다음 중 하나만 결정해라.

- mes: LOT 상태/이력처럼 MES API 호출이 필요한 질문 (예: "ABC0001 상태", "LOT 이력")
- wads: 주간 집계 데이터, 변곡점 분석, Layer1 리포트 관련 질문 (예: "wads 조회", "1월 4일 집계")
- retrieve: 사내 문서/정의/절차 등 검색(RAG)이 필요한 질문 (예: "attention이 뭐야", "공정 절차 알려줘")
- direct: 일반 대화, 인사, 잡담, 간단한 질문 (예: "안녕", "넌 뭐해", "고마워")

중요: "야", "뭐해", "안녕" 같은 짧은 일상 대화는 반드시 direct로 분류해라.

출력은 반드시 한 단어로만: "mes" 또는 "wads" 또는 "retrieve" 또는 "direct"

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
    wads_ctx = state.get("wads_ctx") or {}

    # 짧은 질문(5자 이하)은 LLM 호출 없이 바로 direct로 라우팅
    if len(q) <= 5:
        return "direct"

    # LLM 라우터가 모든 판단을 수행
    decision = route_decider.invoke({"question": q}).strip().lower()

    # 모델 출력 파싱
    if "mes" in decision:
        return "mes"
    if "wads" in decision:
        return "wads"
    if "direct" in decision:
        return "direct"
    if "retrieve" in decision:
        return "retrieve"

    # 기본값: direct (일반 대화로 처리)
    return "direct"


# %%


# langgraph workflow 초기화
workflow = StateGraph(GraphState)

# workflow 노드 추가
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("router_entry", router_entry)
workflow.add_node("llm_answer", llm_answer)
workflow.add_node("mes_agent", mes_agent)
workflow.add_node("wads_agent", wads_agent)

# workflow 엣지 추가
workflow.add_conditional_edges(
    "router_entry",
    route_main,
    {
        "mes": "mes_agent",
        "wads": "wads_agent",
        "retrieve": "retrieve",
        "direct": "llm_answer",
    },
)
workflow.add_edge("retrieve", "llm_answer")
workflow.add_edge("llm_answer", END)
workflow.add_edge("mes_agent", END)
workflow.add_edge("wads_agent", END)

# workflow 진입점 설정
workflow.set_entry_point("router_entry")

# 체크포인터 설정
memory = MemorySaver()

# 컴파일
app = workflow.compile(checkpointer=memory)
app
# # %%
# # 그래프 시각화 및 샘플 실행 (직접 실행 시에만 동작)
# if __name__ == "__main__":
#     from langchain_teddynote.graphs import visualize_graph

#     visualize_graph(app)

#     from langchain_core.runnables import RunnableConfig
#     from langchain_teddynote.messages import invoke_graph, stream_graph, random_uuid

#     # config 설정 (재귀 리밋, thread_id)
#     config = RunnableConfig(
#         recursion_limit=5, configurable={"thread_id": random_uuid()}
#     )

#     # 질문 입력
#     inputs: GraphState = {"question": "attentiom 메커니즘에 대해서 알려줘"}

#     invoke_graph(app, inputs, config)


# %%
