"""OpenRouter LLM 헬퍼 모듈"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)


def get_llm(model_name: str = None, temperature: float = 0) -> ChatOpenAI:
    """OpenRouter API를 사용하는 ChatOpenAI 인스턴스 반환

    Args:
        model_name: 사용할 모델명 (기본값: 환경변수 OPENROUTER_MODEL 또는 gpt-4o-mini)
        temperature: 온도 설정

    Returns:
        ChatOpenAI: OpenRouter 연결된 LLM 인스턴스
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if model_name is None:
        # model_name = os.getenv("OPENROUTER_MODEL", "gpt-oss-120b")
        model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
