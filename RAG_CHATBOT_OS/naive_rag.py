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


# node 정의
from langchain_teddynote.messages import messages_to_history
from func.utils import format_searched_docs


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

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# langgraph workflow 초기화
workflow = StateGraph(GraphState)

# workflow 노드 추가
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("llm_answer", llm_answer)

# workflow 엣지 추가
workflow.add_edge("retrieve", "llm_answer")
workflow.add_edge("llm_answer", END)

# workflow 진입점 설정
workflow.set_entry_point("retrieve")

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
