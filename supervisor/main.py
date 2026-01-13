"""CSV Supervisor Multi-Agent 시스템

Supervisor 패턴으로 CSV Filter Agent와 Visualization Agent를 조정합니다.
"""

from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END, MessagesState

from openrouter_llm import get_llm
from tools import load_and_filter_csv, create_chart
from langfuse_instrumentation import invoke_with_langfuse


# =============================================================================
# 도구 정의 (LangChain Tool로 래핑)
# =============================================================================


@tool
def filter_csv(oper: str = None, para: str = None) -> str:
    """CSV 파일에서 oper, para 조건으로 데이터를 필터링합니다.

    Args:
        oper: 필터링할 oper 값 (예: OP001, OP002)
        para: 필터링할 para 값 (예: TEMP, PRESSURE, HUMIDITY)
    """
    return load_and_filter_csv(oper=oper, para=para)


@tool
def generate_chart(oper: str, para: str, chart_type: str = "line") -> str:
    """시계열 데이터를 차트로 시각화합니다.

    Args:
        oper: 조회할 oper 값 (예: OP001)
        para: 조회할 para 값 (예: TEMP, PRESSURE)
        chart_type: 차트 유형 (line 또는 bar)
    """
    return create_chart(oper=oper, para=para, chart_type=chart_type)


# =============================================================================
# Agent 생성
# =============================================================================


def create_agents():
    """CSV Filter Agent와 Visualization Agent 생성"""

    llm = get_llm()

    # CSV Filter Agent
    filter_agent = create_react_agent(
        model=llm,
        tools=[filter_csv],
        prompt=(
            "당신은 CSV 데이터 필터링 전문 에이전트입니다.\n\n"
            "역할:\n"
            "- filter_data.csv 파일에서 사용자가 요청한 조건으로 데이터를 필터링합니다.\n"
            "- oper(운영코드)와 para(파라미터) 기준으로 필터링할 수 있습니다.\n\n"
            "사용 가능한 값:\n"
            "- oper: OP001, OP002, OP003, OP004, OP005, OP006, OP007\n"
            "- para: TEMP, PRESSURE, HUMIDITY, FLOW, VOLTAGE, CURRENT, VIBRATION, SPEED\n\n"
            "규칙:\n"
            "- 필터링 작업만 수행하세요.\n"
            "- 결과를 명확하게 보고하세요."
        ),
        name="filter_agent",
    )

    # Visualization Agent
    viz_agent = create_react_agent(
        model=llm,
        tools=[generate_chart],
        prompt=(
            "당신은 데이터 시각화 전문 에이전트입니다.\n\n"
            "역할:\n"
            "- timeseries_data.csv 파일의 시계열 데이터를 차트로 시각화합니다.\n"
            "- oper와 para를 지정하여 해당 데이터의 시계열 그래프를 생성합니다.\n\n"
            "사용 가능한 데이터:\n"
            "- OP001: TEMP, PRESSURE\n"
            "- OP002: TEMP, HUMIDITY\n"
            "- OP003: FLOW\n\n"
            "차트 유형:\n"
            "- line: 선 그래프 (기본)\n"
            "- bar: 막대 그래프\n\n"
            "규칙:\n"
            "- 시각화 작업만 수행하세요.\n"
            "- 생성된 차트 파일 경로를 보고하세요."
        ),
        name="viz_agent",
    )

    return filter_agent, viz_agent


# =============================================================================
# Supervisor 그래프 구성
# =============================================================================


def create_supervisor_graph():
    """Supervisor 패턴 멀티에이전트 그래프 생성"""

    llm = get_llm()
    filter_agent, viz_agent = create_agents()

    # Supervisor가 사용할 에이전트 위임 도구
    @tool
    def delegate_to_filter_agent(task: str) -> str:
        """데이터 필터링 작업을 Filter Agent에게 위임합니다.

        Args:
            task: 필터링 작업 설명 (예: "oper가 OP001인 데이터 조회")
        """
        result = filter_agent.invoke({"messages": [{"role": "user", "content": task}]})
        return result["messages"][-1].content

    @tool
    def delegate_to_viz_agent(task: str) -> str:
        """시각화 작업을 Visualization Agent에게 위임합니다.

        Args:
            task: 시각화 작업 설명 (예: "OP001의 TEMP 데이터를 선 그래프로 시각화")
        """
        result = viz_agent.invoke({"messages": [{"role": "user", "content": task}]})
        return result["messages"][-1].content

    # Supervisor Agent 생성
    supervisor = create_react_agent(
        model=llm,
        tools=[delegate_to_filter_agent, delegate_to_viz_agent],
        prompt=(
            "당신은 CSV 데이터 분석 작업을 조정하는 Supervisor입니다.\n\n"
            "관리하는 에이전트:\n"
            "1. Filter Agent: CSV 데이터 필터링 (oper, para 조건)\n"
            "2. Visualization Agent: 시계열 데이터 차트 생성\n\n"
            "작업 분배 규칙:\n"
            "- 데이터 조회/필터링 요청 → Filter Agent\n"
            "- 그래프/차트/시각화 요청 → Visualization Agent\n"
            "- 복합 요청 시 순차적으로 위임\n\n"
            "주의사항:\n"
            "- 직접 작업하지 말고 반드시 에이전트에게 위임하세요.\n"
            "- 에이전트 결과를 종합하여 최종 답변을 제공하세요.\n"
            "- 한국어로 응답하세요."
        ),
        name="supervisor",
    )

    return supervisor


# =============================================================================
# 메인 실행
# =============================================================================


def run_query(query: str):
    """쿼리 실행 및 결과 출력"""
    print(f"\n{'='*60}")
    print(f"📝 질문: {query}")
    print("=" * 60)

    supervisor = create_supervisor_graph()

    # 간단한 태깅(검색/필터 편의용)
    tags = []
    if any(k in query for k in ["연관", "관련", "related"]):
        tags.append("relation")
    if any(k in query for k in ["그래프", "차트", "시각화", "plot", "trend", "시계열"]):
        tags.append("viz")
    if not tags:
        tags.append("filter")

    result = invoke_with_langfuse(supervisor, query, tags=tags)

    # 최종 응답 출력
    final_response = result["messages"][-1].content
    print(f"\n📊 결과:\n{final_response}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    # 테스트 쿼리
    test_queries = [
        "oper가 OP001인 데이터를 보여줘",
        "OP001의 TEMP 값을 시계열 그래프로 그려줘",
        "OP002의 HUMIDITY 데이터를 막대그래프로 시각화해줘",
    ]

    print("\n🤖 CSV Supervisor Multi-Agent 시스템 테스트\n")

    for query in test_queries:
        try:
            run_query(query)
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        print()
