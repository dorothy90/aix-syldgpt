# %%
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os
from func.docs import OpenSearchEmbeddingRetrievalChain
from func.retriever import retriever

# API 키 정보 로드
load_dotenv(override=True)

# OpenSearch 로드합니다.
opensearch = OpenSearchEmbeddingRetrievalChain().create_chain()

# chain을 생성합니다.
chain = opensearch.chain

# %%
# Test
if __name__ == "__main__":
    query = "word2vec이 뭐야"
    # retriever 사용 (하이브리드 검색)
    search_result_retriever = retriever.invoke(query, search_mode="hybrid")
    search_result_chain = chain.invoke(
        {
            "question": query,
            "context": search_result_retriever,
            "chat_history": "",
        }
    )

    print(f"retriever result: {search_result_retriever}")
    print(f"chain result: {search_result_chain}")

# %%

## 랭그래프로 naive rag 구현
# State 정의
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    question: Annotated[str, "Question"]
    context: Annotated[str, "Context"]
    answer: Annotated[str, "Answer"]
    messages: Annotated[list, add_messages]
    # LOT 그래프 관련 필드
    lot_id: Annotated[str, "LOT ID"]
    lot_data: Annotated[dict, "LOT SQL Query Result"]
    lot_plot_image: Annotated[str, "LOT Plot Image (Base64)"]


# node 정의
from langchain_teddynote.messages import messages_to_history
from func.utils import format_searched_docs
from func.lot_utils import extract_lot_id, execute_lot_query, create_lot_plot
from typing import Literal


# 문서 검색 노드
def retrieve_document(state: GraphState) -> GraphState:
    # 질문을 상태에서 가져옴
    latest_question = state["question"]
    # OpenSearch에서 하이브리드 검색으로 관련있는 문서 가져오기
    retrieve_docs = retriever.invoke(latest_question, search_mode="hybrid")
    # 검색된 문서 형식화 (프롬프트에 넣을 때 더 정형화해서 넣기)
    retrieved_docs = format_searched_docs(retrieve_docs)
    # 검색된 문서를 state의 context 키에 저장
    return {"context": retrieved_docs}


# 질문 라우팅 노드
def route_question(state: GraphState) -> Literal["lot_flow", "rag_flow"]:
    """
    질문이 LOT 그래프 요청인지 판별하여 라우팅
    """
    question = state["question"].lower()
    if "lot" in question and ("그려" in question or "그래프" in question or "차트" in question):
        return "lot_flow"
    return "rag_flow"


# LOT 파라미터 추출 노드
def parse_lot_args(state: GraphState) -> GraphState:
    """
    질문에서 LOT ID를 추출합니다.
    """
    question = state["question"]
    lot_id = extract_lot_id(question)

    if not lot_id:
        # LOT ID를 찾을 수 없으면 에러 메시지 반환
        return {
            "answer": "LOT ID를 찾을 수 없습니다. 질문에 LOT 번호를 포함해주세요. 예: 'LOT1234 그려줘'",
            "messages": [("user", question), ("assistant", "LOT ID를 찾을 수 없습니다.")],
        }

    return {"lot_id": lot_id}


# SQL 실행 노드
def run_lot_query(state: GraphState) -> GraphState:
    """
    LOT ID를 기반으로 SQL 쿼리를 실행합니다.
    """
    lot_id = state.get("lot_id", "")
    question = state["question"]

    if not lot_id:
        return {
            "answer": "LOT ID가 없습니다.",
            "messages": [("user", question), ("assistant", "LOT ID가 없습니다.")],
        }

    try:
        df = execute_lot_query(lot_id)

        if df is None or df.empty:
            return {
                "answer": f"{lot_id}에 대한 데이터를 찾을 수 없습니다.",
                "messages": [("user", question), ("assistant", f"{lot_id}에 대한 데이터를 찾을 수 없습니다.")],
            }

        # 데이터프레임을 딕셔너리로 변환 (JSON 직렬화 가능하도록)
        lot_data = {
            "columns": df.columns.tolist(),
            "data": df.values.tolist(),
            "shape": df.shape,
        }

        return {"lot_data": lot_data}

    except Exception as e:
        print(f"SQL 쿼리 실행 오류: {e}")
        return {
            "answer": f"데이터 조회 중 오류가 발생했습니다: {str(e)}",
            "messages": [("user", question), ("assistant", f"데이터 조회 중 오류가 발생했습니다.")],
        }


# 그래프 생성 노드
def generate_lot_plot(state: GraphState) -> GraphState:
    """
    SQL 쿼리 결과를 기반으로 그래프를 생성합니다.
    """
    lot_id = state.get("lot_id", "")
    lot_data = state.get("lot_data", {})
    question = state["question"]

    if not lot_id or not lot_data:
        return {
            "answer": "그래프를 생성할 데이터가 없습니다.",
            "messages": [("user", question), ("assistant", "그래프를 생성할 데이터가 없습니다.")],
        }

    try:
        # 데이터프레임 재구성
        import pandas as pd
        df = pd.DataFrame(lot_data["data"], columns=lot_data["columns"])

        # 그래프 생성
        plot_image = create_lot_plot(df, lot_id)

        if not plot_image:
            return {
                "answer": f"{lot_id} 그래프 생성에 실패했습니다.",
                "messages": [("user", question), ("assistant", f"{lot_id} 그래프 생성에 실패했습니다.")],
            }

        return {"lot_plot_image": plot_image}

    except Exception as e:
        print(f"그래프 생성 오류: {e}")
        return {
            "answer": f"그래프 생성 중 오류가 발생했습니다: {str(e)}",
            "messages": [("user", question), ("assistant", f"그래프 생성 중 오류가 발생했습니다.")],
        }


# LOT 응답 포맷 노드
def lot_response(state: GraphState) -> GraphState:
    """
    LOT 그래프 결과를 최종 응답 형식으로 포맷합니다.
    """
    lot_id = state.get("lot_id", "")
    lot_plot_image = state.get("lot_plot_image", "")
    question = state["question"]

    if lot_plot_image:
        answer = f"{lot_id}에 대한 그래프를 생성했습니다. 아래 이미지를 확인해주세요."
    else:
        answer = f"{lot_id} 그래프 생성에 실패했습니다."

    return {
        "answer": answer,
        "messages": [("user", question), ("assistant", answer)],
    }


# 답변 생성 노드
def llm_answer(state: GraphState) -> GraphState:
    # 질문을 상태에서 가져옴
    latest_question = state["question"]
    # 검색된 문서를 상태에서 가져옴
    context = state["context"]
    # 체인을 호출하여 답변 생성
    response = chain.invoke(
        {
            "question": latest_question,
            "context": context,
            "chat_history": messages_to_history(state["messages"]),
        }
    )
    return {
        "answer": response,
        "messages": [("user", latest_question), ("assistant", response)],
    }


## LangGraph 생성

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# langgraph workflow 초기화
workflow = StateGraph(GraphState)

# workflow 노드 추가
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("llm_answer", llm_answer)
# LOT 관련 노드들
workflow.add_node("parse_lot_args", parse_lot_args)
workflow.add_node("run_lot_query", run_lot_query)
workflow.add_node("generate_lot_plot", generate_lot_plot)
workflow.add_node("lot_response", lot_response)

# workflow 엣지 추가
# START -> lot_flow 또는 rag_flow (조건부 라우팅)
workflow.add_conditional_edges(
    START,
    route_question,  # 라우팅 함수 직접 사용
    {
        "lot_flow": "parse_lot_args",
        "rag_flow": "retrieve",
    }
)

# RAG 플로우
workflow.add_edge("retrieve", "llm_answer")
workflow.add_edge("llm_answer", END)

# LOT 플로우
workflow.add_edge("parse_lot_args", "run_lot_query")
workflow.add_edge("run_lot_query", "generate_lot_plot")
workflow.add_edge("generate_lot_plot", "lot_response")
workflow.add_edge("lot_response", END)

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
    inputs = GraphState(question="attention 메커니즘에 대해서 알려줘")

    invoke_graph(app, inputs, config)


# %%
