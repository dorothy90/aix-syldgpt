# %%
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os
import logging
from func.docs import OpenSearchEmbeddingRetrievalChain
from func.retriever import retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
    api_context: Annotated[str, "API Context"]
    api_documents: Annotated[list, "API Documents"]  # 원본 문서 정보 (filename, url 등)
    opensearch_answer: Annotated[str, "OpenSearch Answer"]
    api_answer: Annotated[str, "API Answer"]
    answer: Annotated[str, "Final Answer"]
    messages: Annotated[list, add_messages]


# node 정의
from langchain_teddynote.messages import messages_to_history
from func.utils import format_searched_docs, format_api_documents
import requests
from typing import List


# 문서 검색 노드 (OpenSearch)
def retrieve_document(state: GraphState) -> GraphState:
    logger.debug("=" * 50)
    logger.debug("[retrieve_document] 노드 시작")
    logger.debug(f"[retrieve_document] 현재 state keys: {list(state.keys())}")

    # 질문을 상태에서 가져옴
    latest_question = state["question"]
    logger.debug(f"[retrieve_document] question: {latest_question}")

    # messages 상태 확인
    messages = state.get("messages", [])
    logger.debug(f"[retrieve_document] messages 개수: {len(messages)}")
    logger.debug(f"[retrieve_document] messages 내용: {messages}")

    try:
        # OpenSearch에서 하이브리드 검색으로 관련있는 문서 가져오기
        retrieve_docs = retriever.invoke(latest_question, search_mode="hybrid")
        # 검색된 문서 형식화 (프롬프트에 넣을 때 더 정형화해서 넣기)
        retrieved_docs = format_searched_docs(retrieve_docs)
        logger.debug(f"[retrieve_document] 검색된 문서 길이: {len(retrieved_docs)}")
        # 검색된 문서를 state의 context 키에 저장
        return {"context": retrieved_docs}
    except Exception as e:
        logger.error(f"[retrieve_document] OpenSearch 검색 오류: {e}")
        return {"context": ""}  # 에러 발생 시 빈 context 반환


# API 문서 검색 노드
def retrieve_api_documents(state: GraphState) -> GraphState:
    """
    외부 API를 통해 문서를 검색하는 노드
    문서를 찾지 못하거나 오류가 발생하면 빈 결과를 반환하여 그래프가 중단되지 않도록 함
    """
    latest_question = state["question"]

    # 환경변수에서 API 설정 가져오기
    api_url = os.getenv("EXTERNAL_API_URL", "")
    api_token = os.getenv("EXTERNAL_API_TOKEN", "")

    # API 설정이 없으면 빈 결과 반환
    if not api_url:
        return {"api_context": "", "api_documents": []}

    try:
        # API 호출
        headers = {
            "Content-Type": "application/json",
        }
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        response = requests.post(
            api_url,
            json={"query": latest_question},
            headers=headers,
            timeout=10,  # 타임아웃 설정
        )

        # 응답 확인
        if response.status_code != 200:
            print(f"API 호출 실패: {response.status_code}")
            return {"api_context": "", "api_documents": []}

        api_data = response.json()

        # API 응답 형식: [{"filename": "...", "content": "...", "url": "..."}, ...]
        api_docs = api_data.get("documents", [])

        if not api_docs or len(api_docs) == 0:
            return {"api_context": "", "api_documents": []}

        # 문서 포맷팅
        formatted_api_docs = format_api_documents(api_docs)

        return {
            "api_context": formatted_api_docs,
            "api_documents": api_docs,  # 원본 문서 정보 저장 (프론트엔드용)
        }

    except requests.exceptions.Timeout:
        print("API 호출 타임아웃")
        return {"api_context": "", "api_documents": []}

    except requests.exceptions.RequestException as e:
        print(f"API 호출 오류: {e}")
        return {"api_context": "", "api_documents": []}

    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        return {"api_context": "", "api_documents": []}


# OpenSearch 기반 답변 생성 노드
def llm_answer_opensearch(state: GraphState) -> GraphState:
    """OpenSearch 검색 결과로 답변 생성 (빈 context도 안전하게 처리)"""
    logger.debug("=" * 50)
    logger.debug("[llm_answer_opensearch] 노드 시작")

    latest_question = state["question"]
    context = state.get("context", "")

    # messages 안전하게 접근 (multiturn 대화 지원)
    messages = state.get("messages", [])
    logger.debug(f"[llm_answer_opensearch] question: {latest_question}")
    logger.debug(f"[llm_answer_opensearch] context 길이: {len(context)}")
    logger.debug(f"[llm_answer_opensearch] messages 개수: {len(messages)}")
    logger.debug(f"[llm_answer_opensearch] messages 타입: {type(messages)}")
    if messages:
        for i, msg in enumerate(messages):
            logger.debug(
                f"[llm_answer_opensearch] message[{i}]: type={type(msg)}, content={msg}"
            )

    # context가 비어있으면 빈 답변 반환
    if not context or context.strip() == "":
        logger.debug("[llm_answer_opensearch] context가 비어있어서 빈 답변 반환")
        return {"opensearch_answer": ""}

    try:
        # chat_history 생성
        chat_history = messages_to_history(messages) if messages else ""
        logger.debug(
            f"[llm_answer_opensearch] chat_history: {chat_history[:200] if chat_history else 'empty'}..."
        )

        response = chain.invoke(
            {
                "question": latest_question,
                "context": context,
                "chat_history": chat_history,
            }
        )
        logger.debug(f"[llm_answer_opensearch] 응답 생성 완료, 길이: {len(response)}")
        return {"opensearch_answer": response}
    except Exception as e:
        logger.error(f"[llm_answer_opensearch] 답변 생성 오류: {e}", exc_info=True)
        return {"opensearch_answer": ""}  # 에러 발생 시 빈 답변 반환


# API 기반 답변 생성 노드
def llm_answer_api(state: GraphState) -> GraphState:
    """API 검색 결과로 답변 생성 (빈 context도 안전하게 처리)"""
    logger.debug("=" * 50)
    logger.debug("[llm_answer_api] 노드 시작")

    latest_question = state["question"]
    api_context = state.get("api_context", "")

    # messages 안전하게 접근 (multiturn 대화 지원)
    messages = state.get("messages", [])
    logger.debug(f"[llm_answer_api] question: {latest_question}")
    logger.debug(f"[llm_answer_api] api_context 길이: {len(api_context)}")
    logger.debug(f"[llm_answer_api] messages 개수: {len(messages)}")

    # api_context가 비어있으면 빈 답변 반환
    if not api_context or api_context.strip() == "":
        logger.debug("[llm_answer_api] api_context가 비어있어서 빈 답변 반환")
        return {"api_answer": ""}

    try:
        # chat_history 생성
        chat_history = messages_to_history(messages) if messages else ""
        logger.debug(
            f"[llm_answer_api] chat_history: {chat_history[:200] if chat_history else 'empty'}..."
        )

        response = chain.invoke(
            {
                "question": latest_question,
                "context": api_context,
                "chat_history": chat_history,
            }
        )
        logger.debug(f"[llm_answer_api] 응답 생성 완료, 길이: {len(response)}")
        return {"api_answer": response}
    except Exception as e:
        logger.error(f"[llm_answer_api] 답변 생성 오류: {e}", exc_info=True)
        return {"api_answer": ""}  # 에러 발생 시 빈 답변 반환


# 두 답변을 합치는 노드
def merge_answers(state: GraphState) -> GraphState:
    """두 답변을 LLM을 사용하여 종합하고 자연스럽게 연결"""
    logger.debug("=" * 50)
    logger.debug("[merge_answers] 노드 시작")

    opensearch_answer = state.get("opensearch_answer", "")
    api_answer = state.get("api_answer", "")
    api_documents = state.get("api_documents", [])  # API 문서 메타데이터 가져오기
    latest_question = state["question"]

    # messages 상태 확인
    messages = state.get("messages", [])
    logger.debug(f"[merge_answers] question: {latest_question}")
    logger.debug(f"[merge_answers] opensearch_answer 길이: {len(opensearch_answer)}")
    logger.debug(f"[merge_answers] api_answer 길이: {len(api_answer)}")
    logger.debug(f"[merge_answers] 기존 messages 개수: {len(messages)}")

    # 둘 다 답변이 없는 경우
    if not opensearch_answer and not api_answer:
        error_msg = "관련된 정보를 찾을 수 없습니다."
        logger.debug("[merge_answers] 두 답변 모두 비어있음, 에러 메시지 반환")
        return {
            "answer": error_msg,
            "messages": [
                HumanMessage(content=latest_question),
                AIMessage(content=error_msg),
            ],
        }

    # API 문서의 URL 정보 추출
    api_urls = []
    if api_documents:
        for doc in api_documents:
            url = doc.get("url", "")
            filename = doc.get("filename", "")
            if url:
                api_urls.append(f"- {filename}: {url}" if filename else f"- {url}")

    # URL 정보를 포함한 프롬프트 생성
    url_section = ""
    if api_urls:
        url_section = (
            f"\n\n[참고 문서 링크]\n"
            + "\n".join(api_urls)
            + "\n\n위 링크들을 답변에 자연스럽게 포함시켜주세요."
        )

    # 병합 프롬프트 생성
    merge_prompt = PromptTemplate.from_template(
        """질문: {question}

다음 두 가지 출처의 답변을 종합하여 하나의 완성된 답변을 작성해주세요.
중복된 내용은 통합하고, 서로 보완되는 내용은 자연스럽게 연결해주세요.
각 출처의 정보가 명확히 구분되도록 하되, 전체적인 흐름이 끊기지 않도록 해주세요.

[출처 1: 내부 문서]
{answer1}

[출처 2: 외부 API]
{answer2}
{url_section}

종합 답변:"""
    )

    # LLM 체인 실행
    # opensearch 객체에서 모델 생성
    llm = opensearch.create_model()
    merge_chain = merge_prompt | llm | StrOutputParser()

    response = merge_chain.invoke(
        {
            "question": latest_question,
            "answer1": opensearch_answer if opensearch_answer else "내용 없음",
            "answer2": api_answer if api_answer else "내용 없음",
            "url_section": url_section,
        }
    )

    logger.debug(f"[merge_answers] 최종 응답 생성 완료, 길이: {len(response)}")
    logger.debug(f"[merge_answers] messages에 추가: HumanMessage + AIMessage")

    return {
        "answer": response,
        "messages": [
            HumanMessage(content=latest_question),
            AIMessage(content=response),
        ],
    }


## LangGraph 생성

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# langgraph workflow 초기화
workflow = StateGraph(GraphState)

# workflow 노드 추가
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("retrieve_api", retrieve_api_documents)
workflow.add_node("llm_answer_opensearch", llm_answer_opensearch)
workflow.add_node("llm_answer_api", llm_answer_api)
workflow.add_node("merge_answers", merge_answers)


# 병렬 실행을 위한 라우팅 함수
def route_to_retrievers(state: GraphState) -> List[str]:
    """retrieve와 retrieve_api를 병렬로 실행"""
    return ["retrieve", "retrieve_api"]


# START에서 병렬로 시작
workflow.add_conditional_edges(START, route_to_retrievers, ["retrieve", "retrieve_api"])

# 각 검색 후 해당 llm_answer로
workflow.add_edge("retrieve", "llm_answer_opensearch")
workflow.add_edge("retrieve_api", "llm_answer_api")

# 두 답변 생성 후 merge로 (병렬 실행)
workflow.add_edge("llm_answer_opensearch", "merge_answers")
workflow.add_edge("llm_answer_api", "merge_answers")

# 최종 답변 후 종료
workflow.add_edge("merge_answers", END)

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
    # 중요: multiturn 대화를 위해 동일한 thread_id 유지
    thread_id = random_uuid()
    logger.info(f"[테스트] thread_id: {thread_id}")

    config = RunnableConfig(recursion_limit=5, configurable={"thread_id": thread_id})

    # === 첫 번째 질문 ===
    logger.info("=" * 60)
    logger.info("[테스트] 첫 번째 질문 시작")
    inputs = {"question": "attention 메커니즘에 대해서 알려줘", "messages": []}

    result1 = app.invoke(inputs, config=config)
    logger.info(f"[테스트] 첫 번째 응답: {result1.get('answer', '')[:200]}...")
    logger.info(f"[테스트] 현재 messages 개수: {len(result1.get('messages', []))}")

    # === 두 번째 질문 (follow-up) ===
    logger.info("=" * 60)
    logger.info("[테스트] 두 번째 질문 시작 (멀티턴 테스트)")
    inputs2 = {"question": "그게 트랜스포머에서 어떻게 사용돼?", "messages": []}

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
            content_preview = (
                str(msg.content)[:100] if hasattr(msg, "content") else str(msg)[:100]
            )
            logger.info(f"[테스트] message[{i}]: {msg_type} - {content_preview}...")


# %%
