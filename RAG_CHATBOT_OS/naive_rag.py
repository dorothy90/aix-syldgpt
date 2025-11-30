# %%
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os
import logging
from func.docs import OpenSearchEmbeddingRetrievalChain
from func.retriever import retriever
from langchain_core.messages import HumanMessage, AIMessage

# 디버깅 로거 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    logger.debug("=" * 50)
    logger.debug("[retrieve_document] 노드 시작")
    logger.debug(f"[retrieve_document] state keys: {list(state.keys())}")

    # 질문을 상태에서 가져옴
    latest_question = state["question"]
    logger.debug(f"[retrieve_document] question: {latest_question}")

    # messages 상태 확인 (디버깅용)
    messages = state.get("messages", [])
    logger.debug(f"[retrieve_document] messages 개수: {len(messages)}")
    if messages:
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content = msg.content if hasattr(msg, "content") else str(msg)
            logger.debug(
                f"[retrieve_document] message[{i}]: {msg_type} - {content[:50]}..."
            )

    # OpenSearch에서 하이브리드 검색으로 관련있는 문서 가져오기
    retrieve_docs = retriever.invoke(latest_question, search_mode="hybrid")
    # 검색된 문서 형식화 (프롬프트에 넣을 때 더 정형화해서 넣기)
    retrieved_docs = format_searched_docs(retrieve_docs)
    logger.debug(f"[retrieve_document] 검색된 문서 길이: {len(retrieved_docs)}")

    # 검색된 문서를 state의 context 키에 저장
    return {"context": retrieved_docs}


# 답변 생성 노드
def llm_answer(state: GraphState) -> GraphState:
    logger.debug("=" * 50)
    logger.debug("[llm_answer] 노드 시작")

    # 질문을 상태에서 가져옴
    latest_question = state["question"]
    logger.debug(f"[llm_answer] question: {latest_question}")

    # 검색된 문서를 상태에서 가져옴
    context = state.get("context", "")
    logger.debug(f"[llm_answer] context 길이: {len(context)}")

    # messages 안전하게 접근 (멀티턴 대화 지원)
    messages = state.get("messages", [])
    logger.debug(f"[llm_answer] messages 개수: {len(messages)}")
    logger.debug(f"[llm_answer] messages 타입: {type(messages)}")
    if messages:
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content = msg.content if hasattr(msg, "content") else str(msg)
            logger.debug(f"[llm_answer] message[{i}]: {msg_type} - {content[:100]}...")

    # chat_history 생성 (messages가 있을 때만)
    chat_history = messages_to_history(messages) if messages else ""
    logger.debug(
        f"[llm_answer] chat_history: {chat_history[:200] if chat_history else 'empty'}..."
    )

    # 체인을 호출하여 답변 생성
    response = chain.invoke(
        {
            "question": latest_question,
            "context": context,
            "chat_history": chat_history,
        }
    )
    logger.debug(f"[llm_answer] 응답 생성 완료, 길이: {len(response)}")

    # HumanMessage, AIMessage 형식으로 반환 (add_messages reducer 호환)
    return {
        "answer": response,
        "messages": [
            HumanMessage(content=latest_question),
            AIMessage(content=response),
        ],
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
    from langchain_teddynote.messages import random_uuid

    # config 설정 (재귀 리밋, thread_id)
    # 중요: 멀티턴 대화를 위해 동일한 thread_id 유지
    thread_id = random_uuid()
    logger.info(f"[테스트] thread_id: {thread_id}")

    config = RunnableConfig(recursion_limit=5, configurable={"thread_id": thread_id})

    # === 첫 번째 질문 ===
    logger.info("=" * 60)
    logger.info("[테스트] 첫 번째 질문 시작")
    inputs = {"question": "attention 메커니즘에 대해서 알려줘"}

    result1 = app.invoke(inputs, config=config)
    logger.info(f"[테스트] 첫 번째 응답: {result1.get('answer', '')[:200]}...")
    logger.info(f"[테스트] 현재 messages 개수: {len(result1.get('messages', []))}")

    # === 두 번째 질문 (멀티턴 테스트) ===
    logger.info("=" * 60)
    logger.info("[테스트] 두 번째 질문 시작 (멀티턴 테스트)")
    inputs2 = {"question": "그게 트랜스포머에서 어떻게 사용돼?"}

    result2 = app.invoke(inputs2, config=config)
    logger.info(f"[테스트] 두 번째 응답: {result2.get('answer', '')[:200]}...")
    logger.info(f"[테스트] 현재 messages 개수: {len(result2.get('messages', []))}")

    # === 최종 상태 확인 ===
    logger.info("=" * 60)
    logger.info("[테스트] 최종 상태 확인")
    final_state = app.get_state(config)
    if hasattr(final_state, "values"):
        messages = final_state.values.get("messages", [])
        logger.info(f"[테스트] 저장된 전체 messages 개수: {len(messages)}")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content = msg.content[:100] if hasattr(msg, "content") else str(msg)[:100]
            logger.info(f"[테스트] message[{i}]: {msg_type} - {content}...")


# %%
