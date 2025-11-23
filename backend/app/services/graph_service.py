import sys
from pathlib import Path
from typing import AsyncIterator
from langchain_core.runnables import RunnableConfig

# RAG_CHATBOT_OS 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
rag_chatbot_os_path = project_root / "RAG_CHATBOT_OS"
sys.path.insert(0, str(rag_chatbot_os_path))


class GraphService:
    """LangGraph 서비스 래퍼"""

    def __init__(self):
        self._app = None
        self._GraphState = None

    @property
    def app(self):
        """Lazy import: 실제 사용 시에만 naive_rag를 import"""
        if self._app is None:
            # 기존 naive_rag 모듈에서 컴파일된 그래프 import
            from naive_rag import app as graph_app, GraphState

            self._app = graph_app
            self._GraphState = GraphState
        return self._app

    @property
    def GraphState(self):
        """GraphState 타입 반환"""
        if self._GraphState is None:
            from naive_rag import GraphState

            self._GraphState = GraphState
        return self._GraphState

    async def stream(self, question: str, config: RunnableConfig) -> AsyncIterator[str]:
        """
        스트리밍 방식으로 LangGraph 실행

        Args:
            question: 사용자 질문
            config: RunnableConfig (thread_id 포함)

        Yields:
            str: 스트리밍된 토큰들
        """
        # GraphState는 TypedDict이므로 딕셔너리로 사용
        inputs = {"question": question}

        # stream_mode="messages"로 LLM 토큰 단위 스트리밍
        # 반환 형태: (chunk_msg, metadata) 튜플
        async for chunk_msg, metadata in self.app.astream(
            inputs, config=config, stream_mode="messages"
        ):
            # llm_answer 노드에서 출력된 메시지만 스트리밍
            if metadata.get("langgraph_node") == "llm_answer":
                if hasattr(chunk_msg, "content") and chunk_msg.content:
                    yield chunk_msg.content

    def invoke(self, question: str, config: RunnableConfig) -> dict:
        """
        일반 방식으로 LangGraph 실행 (비스트리밍)

        Args:
            question: 사용자 질문
            config: RunnableConfig (thread_id 포함)

        Returns:
            dict: 실행 결과
        """
        # GraphState는 TypedDict이므로 딕셔너리로 사용
        inputs = {"question": question}

        return self.app.invoke(inputs, config=config)

    def get_state(self, config: RunnableConfig) -> dict:
        """
        현재 thread_id의 최종 state를 가져옴

        Args:
            config: RunnableConfig (thread_id 포함)

        Returns:
            dict: 현재 state
        """
        # get_state를 사용하여 현재 state 가져오기
        state = self.app.get_state(config)
        return state.values if hasattr(state, "values") else {}


# 싱글톤 인스턴스
graph_service = GraphService()
