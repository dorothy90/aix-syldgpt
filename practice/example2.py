# repl 사용, graph는 팝업으로 띄우는 버젼

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_experimental.tools import PythonREPLTool
from dotenv import load_dotenv
import os
from typing import TypedDict, Optional
import base64

# API KEY 정보로드
load_dotenv(override=True)
# GPT-4.1 모델 초기화
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.7,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)

repl = PythonREPLTool()
# --- 3️⃣ System Prompt 정의 ---
basic_chatbot_prompt = """
You are the Basic Chatbot.
- Handle general conversation.
- If the question involves code, route to the Code Agent.
- If the question involves charts or visualization, route to the Visualization Agent.
"""

code_agent_prompt = """
You are the Code Agent.
- Generate Python code for data analysis or computation using pandas, numpy, or scipy.
- Include clear comments.
- Do NOT create visualizations or charts.
"""
viz_agent_prompt = """
You are the Visualization Agent.
Generate Python code using matplotlib or seaborn to visualize the data.
- Include plt.show() at the end.
- DO NOT describe the result, only provide code.
"""


# --- 1. 상태 스키마 정의 ---
class AgentState(TypedDict):
    input: str
    route: Optional[str]
    output: Optional[str]


# --- Step 1: Node 정의 ---
def basic_chatbot_node(state: AgentState) -> AgentState:
    """사용자 요청을 분류하고 route 지정"""
    user_input = state.get("input", "").lower()

    if any(
        k in user_input
        for k in ["plot", "그래프", "시각화", "boxplot", "scatter", "curve"]
    ):
        route = "viz_agent"
    elif any(k in user_input for k in ["code", "코드", "python", "함수", "데이터분석"]):
        route = "code_agent"
    else:
        route = "basic_chatbot"

    # route를 항상 반환해야 함
    return {"input": state["input"], "route": route, "output": None}


def code_agent_node(state: AgentState) -> AgentState:
    messages = [
        {"role": "system", "content": code_agent_prompt},
        {"role": "user", "content": state["input"]},
    ]
    response = llm.invoke(messages)
    return {"input": state["input"], "route": "code_agent", "output": response.content}


import re


def viz_agent_node(state: AgentState) -> AgentState:
    """시각화 코드 생성 + 자동 실행 + 이미지 저장 + base64 인코딩"""
    messages = [
        {"role": "system", "content": viz_agent_prompt},
        {"role": "user", "content": state["input"]},
    ]
    response = llm.invoke(messages)
    raw_text = response.content

    # --- 1️⃣ 코드 블록만 추출 ---
    code_match = re.search(r"```(?:python)?\n([\s\S]*?)```", raw_text)
    if code_match:
        clean_code = code_match.group(1).strip()
    else:
        clean_code = re.split(r"\*\*설명", raw_text, 1)[0].strip()

    # --- 2️⃣ Markdown, 설명 제거 ---
    clean_code = re.sub(r"\*\*.*?\*\*", "", clean_code)
    clean_code = re.sub(r"\$.*?\$", "", clean_code)
    clean_code = clean_code.strip()

    # --- 3️⃣ 자동 저장 코드 삽입 ---
    save_path = "/tmp/plot.png"
    clean_code = (
        clean_code + f"\nplt.savefig('{save_path}')\n"
        "print('✅ 그래프 저장 완료:', '" + save_path + "')\n"
        "plt.close()\n"
    )

    print("\n--- [Generated Code] ---\n")
    print(clean_code)

    # --- 4️⃣ 실행 ---
    try:
        result = repl.run(clean_code)
        print("\n--- [Execution Result] ---\n")
        print(result)

        # --- 5️⃣ 이미지 base64 인코딩 ---
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            print(f"🖼  그래프 이미지 저장됨: {save_path}")
        else:
            image_b64 = None

    except Exception as e:
        result = f"Error while executing: {e}"
        image_b64 = None

    return {
        "input": state["input"],
        "route": "viz_agent",
        "output": result,
        "image_b64": image_b64,
    }


# --- 5️⃣ Graph 구성 ---
graph = StateGraph(AgentState)

graph.add_node("basic_chatbot", basic_chatbot_node)
graph.add_node("code_agent", code_agent_node)
graph.add_node("viz_agent", viz_agent_node)

# 시작점 → basic_chatbot
graph.add_edge(START, "basic_chatbot")

# 라우팅 분기
graph.add_conditional_edges(
    "basic_chatbot",
    lambda s: s.get("route", "basic_chatbot"),
    {"code_agent": "code_agent", "viz_agent": "viz_agent", "basic_chatbot": END},
)

# 각 agent 종료
graph.add_edge("code_agent", END)
graph.add_edge("viz_agent", END)

# --- 6️⃣ 그래프 컴파일 ---
app = graph.compile()

# --- 7️⃣ 실행 예시 ---
response = app.invoke({"input": "sin 그래프 그려줘"})
print("\n[Assistant Output]\n", response["output"])
