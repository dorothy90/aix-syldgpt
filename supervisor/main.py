"""CSV Supervisor Multi-Agent 시스템

Supervisor 패턴으로 CSV Filter Agent와 Visualization Agent를 조정합니다.
"""

from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END, MessagesState

from openrouter_llm import get_llm
from tools import load_and_filter_csv, create_chart


# =============================================================================
# 도구 정의 (LangChain Tool로 래핑)
# =============================================================================


@tool
def filter_csv(oper: str = None, para: str = None) -> str:
    """CSV 파일에서 oper, para 조건으로 데이터를 필터링합니다.

    Args:
        oper: 필터링할 공정코드 (예: GT PLUG ETCH CD, BLC ETCH CD HV)
        para: 필터링할 파라미터 (예: CD_TOP, DEPTH, TEMP)
    """
    # #region agent log
    import json

    open(
        "/Users/daehwankim/Documents/langgraph-tutorial-main/.cursor/debug.log", "a"
    ).write(
        json.dumps(
            {
                "location": "main.py:filter_csv",
                "message": "filter_csv TOOL CALLED",
                "data": {"oper": oper, "para": para},
                "hypothesisId": "B",
                "timestamp": __import__("time").time(),
            }
        )
        + "\n"
    )
    # #endregion
    return load_and_filter_csv(oper=oper, para=para)


@tool
def generate_chart(oper: str, para: str, chart_type: str = "line") -> str:
    """시계열 데이터를 차트로 시각화합니다.

    Args:
        oper: 조회할 공정코드 (예: GT PLUG ETCH CD, BLC ETCH CD HV)
        para: 조회할 파라미터 (예: CD_TOP, DEPTH, TEMP)
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
            "- oper(공정코드)와 para(파라미터) 기준으로 필터링할 수 있습니다.\n\n"
            "사용 가능한 oper 예시 (반도체 공정명):\n"
            "- GT PLUG ETCH CD, BLC ETCH CD HV, STI CMP THK, METAL CVD RATE\n"
            "- GATE OX THK, SPACER DEP CD, ILD CMP UNIF, CONTACT ETCH DEPTH\n"
            "- PMD DEP RATE, BARRIER ALD THK, W PLUG FILL, POLY ETCH CD\n"
            "- HM STRIP RATE, DIFF ANNEAL TEMP, ION IMP DOSE\n\n"
            "사용 가능한 para 예시:\n"
            "- CD_TOP, CD_BTM, CD_MID, CD, DEPTH, THK, TEMP, PRESSURE\n"
            "- UNIFORMITY, DEP_RATE, REMOVAL_RATE, VOLTAGE, CURRENT 등\n\n"
            "규칙:\n"
            "- 사용자가 요청한 값을 그대로 사용하여 필터링하세요.\n"
            "- 반드시 filter_csv 도구를 호출하여 실제 데이터를 조회하세요.\n"
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
            "- oper(공정코드)와 para(파라미터)를 지정하여 시계열 그래프를 생성합니다.\n\n"
            "사용 가능한 oper 예시 (반도체 공정명):\n"
            "- GT PLUG ETCH CD, BLC ETCH CD HV, STI CMP THK, METAL CVD RATE 등\n\n"
            "사용 가능한 para 예시:\n"
            "- CD_TOP, CD_BTM, DEPTH, THK, TEMP, PRESSURE, UNIFORMITY 등\n\n"
            "차트 유형:\n"
            "- line: 선 그래프 (기본)\n"
            "- bar: 막대 그래프\n\n"
            "규칙:\n"
            "- 사용자가 요청한 값을 그대로 사용하여 차트를 생성하세요.\n"
            "- 반드시 generate_chart 도구를 호출하세요.\n"
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

    result = supervisor.invoke({"messages": [{"role": "user", "content": query}]})

    # 최종 응답 출력
    final_response = result["messages"][-1].content
    print(f"\n📊 결과:\n{final_response}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    # 테스트 쿼리
    test_queries = [
        "oper가 GT PLUG ETCH CD인 데이터를 보여줘",
        # "OP001의 TEMP 값을 시계열 그래프로 그려줘",
        # "OP002의 HUMIDITY 데이터를 막대그래프로 시각화해줘",
    ]

    print("\n🤖 CSV Supervisor Multi-Agent 시스템 테스트\n")

    for query in test_queries:
        try:
            run_query(query)
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        print()
