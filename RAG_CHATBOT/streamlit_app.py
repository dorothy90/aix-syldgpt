import streamlit as st
from uuid import uuid4
from langchain_core.runnables import RunnableConfig
from datetime import datetime
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Dict, List

# 환경변수 로드
load_dotenv(override=True)

# 같은 디렉터리의 naive_rag 모듈에서 컴파일된 그래프(app)와 상태 타입을 불러옵니다.
from naive_rag import app as graph_app, GraphState


st.set_page_config(page_title="AIME Assistant", page_icon="💬", layout="centered")


# MongoDB 연결
@st.cache_resource
def get_mongo_client():
    """MongoDB 클라이언트 반환 (캐시됨)"""
    return MongoClient("mongodb://localhost:27017/")


def get_sessions_collection():
    """채팅 세션 컬렉션 반환"""
    client = get_mongo_client()
    db = client["chat_history"]
    return db["sessions"]


def get_feedback_collection():
    """피드백 컬렉션 반환"""
    client = get_mongo_client()
    db = client["chat_history"]
    return db["feedbacks"]


def load_sessions_from_db() -> Dict:
    """MongoDB에서 모든 세션 로드"""
    try:
        collection = get_sessions_collection()
        sessions = {}
        for doc in collection.find():
            session_id = doc["session_id"]
            sessions[session_id] = {
                "title": doc.get("title", "새 채팅"),
                "messages": doc.get("messages", []),
                "created_at": doc.get(
                    "created_at", datetime.now().strftime("%Y-%m-%d %H:%M")
                ),
            }
        return sessions
    except Exception as e:
        print(f"세션 로드 실패: {e}")
        return {}


def save_session_to_db(session_id: str, session_data: Dict) -> None:
    """세션을 MongoDB에 저장 (생성 또는 업데이트)"""
    try:
        collection = get_sessions_collection()
        doc = {
            "session_id": session_id,
            "title": session_data["title"],
            "messages": session_data["messages"],
            "created_at": session_data["created_at"],
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # upsert: 있으면 업데이트, 없으면 생성
        collection.update_one({"session_id": session_id}, {"$set": doc}, upsert=True)
    except Exception as e:
        print(f"세션 저장 실패: {e}")


def delete_session_from_db(session_id: str) -> None:
    """MongoDB에서 세션 삭제"""
    try:
        collection = get_sessions_collection()
        collection.delete_one({"session_id": session_id})
    except Exception as e:
        print(f"세션 삭제 실패: {e}")


def save_feedback_to_db(user_name: str, feedback_content: str) -> bool:
    """피드백을 MongoDB에 저장"""
    try:
        collection = get_feedback_collection()
        feedback_doc = {
            "user_name": user_name,
            "feedback_content": feedback_content,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": (
                get_current_session_id()
                if "current_session_id" in st.session_state
                else None
            ),
        }
        collection.insert_one(feedback_doc)
        return True
    except Exception as e:
        print(f"피드백 저장 실패: {e}")
        return False


# LLM 초기화 (제목 생성용)
@st.cache_resource
def get_title_generator_llm():
    """제목 생성용 LLM 인스턴스 반환 (캐시됨)"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    model_name = os.getenv("RETRIEVE_CHAIN_MODEL")
    return ChatOpenAI(
        model_name=model_name,
        temperature=0.3,
        api_key=api_key,
        base_url=base_url,
    )


def generate_chat_title(user_message: str, assistant_message: str = None) -> str:
    """LLM을 사용하여 대화 내용을 요약한 제목 생성"""
    try:
        llm = get_title_generator_llm()

        # 프롬프트 구성
        if assistant_message:
            content = f"""다음 대화 내용을 보고, 이 채팅 세션의 제목을 짧고 간결하게 생성해주세요.

사용자: {user_message}
어시스턴트: {assistant_message}

요구사항:
- 20자 이내로 작성
- 대화의 핵심 주제를 포함
- 명확하고 구체적으로
- 이모지 없이 텍스트만

제목:"""
        else:
            content = f"""다음 사용자 질문을 보고, 이 채팅 세션의 제목을 짧고 간결하게 생성해주세요.

사용자: {user_message}

요구사항:
- 20자 이내로 작성
- 질문의 핵심 주제를 포함
- 명확하고 구체적으로
- 이모지 없이 텍스트만

제목:"""

        # LLM 호출
        response = llm.invoke(content)
        title = response.content.strip()

        # 제목이 너무 길면 자르기
        if len(title) > 25:
            title = title[:25] + "..."

        return title
    except Exception as e:
        print(f"제목 생성 실패: {e}")
        # 실패시 첫 메시지의 일부를 사용
        return user_message[:20] + "..." if len(user_message) > 20 else user_message


def init_sessions() -> None:
    """세션 목록 초기화 - MongoDB에서 로드"""
    if "sessions" not in st.session_state:
        # MongoDB에서 기존 세션 로드
        st.session_state.sessions = load_sessions_from_db()

        # 세션이 없으면 새로 생성
        if not st.session_state.sessions:
            create_new_session()

    if "current_session_id" not in st.session_state:
        # 가장 최근 세션을 현재 세션으로 설정
        if st.session_state.sessions:
            sorted_sessions = sorted(
                st.session_state.sessions.items(),
                key=lambda x: x[1]["created_at"],
                reverse=True,
            )
            st.session_state.current_session_id = sorted_sessions[0][0]
        else:
            create_new_session()


def create_new_session() -> str:
    """새로운 채팅 세션 생성 및 MongoDB에 저장"""
    session_id = str(uuid4())
    session_data = {
        "title": f"새 채팅 {len(st.session_state.sessions) + 1}",
        "messages": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state.sessions[session_id] = session_data
    st.session_state.current_session_id = session_id

    # MongoDB에 저장
    save_session_to_db(session_id, session_data)

    return session_id


def get_current_session_id() -> str:
    """현재 활성 세션 ID 반환"""
    if "current_session_id" not in st.session_state:
        init_sessions()
    return st.session_state.current_session_id


def get_current_messages() -> list:
    """현재 세션의 메시지 목록 반환"""
    session_id = get_current_session_id()
    return st.session_state.sessions[session_id]["messages"]


def update_session_title(
    session_id: str, user_message: str = None, assistant_message: str = None
) -> None:
    """LLM을 사용하여 세션 제목을 자동 생성 및 업데이트 후 MongoDB에 저장"""
    if st.session_state.sessions[session_id]["title"].startswith("새 채팅"):
        if user_message:
            # LLM을 사용하여 제목 생성
            with st.spinner("제목 생성 중..."):
                title = generate_chat_title(user_message, assistant_message)
            st.session_state.sessions[session_id]["title"] = title

            # MongoDB에 업데이트
            save_session_to_db(session_id, st.session_state.sessions[session_id])


def init_chat_history() -> None:
    """채팅 히스토리 초기화 (하위 호환성 유지)"""
    init_sessions()


def render_history() -> None:
    """현재 세션의 메시지 히스토리 렌더링"""
    messages = get_current_messages()
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


def render_sidebar() -> None:
    """사이드바 렌더링 - 피드백, 새 채팅 버튼과 채팅 히스토리"""
    with st.sidebar:
        st.title("💬 채팅 관리")

        # 피드백 섹션
        with st.expander("📝 피드백 보내기", expanded=False):
            st.caption("여러분의 소중한 의견을 들려주세요!")

            # 폼을 사용하여 제출 후 자동 초기화
            with st.form(key="feedback_form", clear_on_submit=True):
                # 이용자 입력 필드
                user_name = st.text_input(
                    "사번",
                    placeholder="예: 홍길동 또는 user@example.com",
                )

                # 피드백 내용 입력 필드
                feedback_content = st.text_area(
                    "Feedback",
                    placeholder="개선 사항, 버그 제보, 건의사항 등을 자유롭게 작성해주세요.",
                    height=100,
                )

                # 보내기 버튼
                submit_button = st.form_submit_button(
                    "📤 피드백 보내기", use_container_width=True
                )

            # 폼 제출 처리
            if submit_button:
                if not user_name or not user_name.strip():
                    st.warning("⚠️ 이름 또는 이메일을 입력해주세요.")
                elif not feedback_content or not feedback_content.strip():
                    st.warning("⚠️ 피드백 내용을 입력해주세요.")
                else:
                    # MongoDB에 피드백 저장
                    if save_feedback_to_db(user_name.strip(), feedback_content.strip()):
                        st.success("✅ 피드백이 성공적으로 전송되었습니다. 감사합니다!")
                    else:
                        st.error("❌ 피드백 전송에 실패했습니다. 다시 시도해주세요.")

        st.divider()

        # 새 채팅 버튼
        if st.button(
            "➕ 새 채팅",
            use_container_width=True,
        ):
            create_new_session()
            st.rerun()

        st.divider()

        # 채팅 히스토리
        st.subheader("📝 채팅 히스토리")

        if st.session_state.sessions:
            # 세션을 생성 시간 역순으로 정렬
            sorted_sessions = sorted(
                st.session_state.sessions.items(),
                key=lambda x: x[1]["created_at"],
                reverse=True,
            )

            for session_id, session_data in sorted_sessions:
                col1, col2 = st.columns([7, 0.1])

                with col1:
                    # 현재 활성 세션 표시
                    is_current = session_id == get_current_session_id()
                    button_type = "primary" if is_current else "secondary"

                    if st.button(
                        f"{''if is_current else ''}{session_data['title']}",
                        key=f"session_{session_id}",
                        use_container_width=True,
                    ):
                        if not is_current:
                            st.session_state.current_session_id = session_id
                            st.rerun()

                # with col2:
                # 삭제 버튼
                # if st.button("삭제", key=f"delete_{session_id}"):
                #     if len(st.session_state.sessions) > 1:
                #         # 메모리에서 삭제
                #         del st.session_state.sessions[session_id]
                #         # MongoDB에서 삭제
                #         delete_session_from_db(session_id)
                #         # 삭제된 세션이 현재 세션이면 다른 세션으로 전환
                #         if session_id == get_current_session_id():
                #             st.session_state.current_session_id = list(
                #                 st.session_state.sessions.keys()
                #             )[0]
                #         st.rerun()
                #     else:
                #         st.warning("마지막 채팅은 삭제할 수 없습니다.")

                # 생성 시간 표시
                # st.caption(f"📅 {session_data['created_at']}")
                # st.divider()
        else:
            st.info("채팅 히스토리가 없습니다.")


def ask_and_answer(user_text: str) -> None:
    """사용자 질문을 받아 LangGraph로 답변 생성 및 MongoDB에 저장"""
    session_id = get_current_session_id()
    messages = get_current_messages()

    # 첫 메시지인지 확인
    is_first_message = len(messages) == 0

    # 사용자 메시지 추가 및 출력
    messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    # LangGraph 실행
    try:
        config = RunnableConfig(
            recursion_limit=10,
            configurable={"thread_id": session_id},  # 세션 ID를 thread_id로 사용
        )

        # 첫 번째 메시지인 경우에만 빈 messages로 시작
        # 이후에는 체크포인터가 자동으로 이전 대화를 불러옴
        inputs: GraphState = {
            "question": user_text,
        }

        result = graph_app.invoke(inputs, config=config)
        answer = result.get("answer", "") if isinstance(result, dict) else result
    except Exception as e:
        st.error(f"그래프 실행 중 오류가 발생했습니다: {e}")
        return

    # 어시스턴트 메시지 추가 및 출력
    messages.append({"role": "assistant", "content": str(answer)})
    with st.chat_message("assistant"):
        st.write(str(answer))

    # MongoDB에 메시지 업데이트 저장
    save_session_to_db(session_id, st.session_state.sessions[session_id])

    # 첫 대화 완료 후 LLM을 사용하여 세션 제목 업데이트 (이미 save_session_to_db 포함)
    if is_first_message:
        update_session_title(session_id, user_text, str(answer))


def main() -> None:
    """메인 함수"""
    # 세션 초기화
    init_chat_history()

    # 사이드바 렌더링
    render_sidebar()

    # 메인 타이틀
    st.title("🤖 AIME Assistant")

    # 현재 세션 정보 표시
    current_session = st.session_state.sessions[get_current_session_id()]
    st.caption(f"💬 {current_session['title']} | 📅 {current_session['created_at']}")

    st.divider()

    # 채팅 히스토리 렌더링
    render_history()

    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        ask_and_answer(prompt)


if __name__ == "__main__":
    main()
