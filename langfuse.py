#%%
from langfuse import Langfuse
from openai import OpenAI

lf = Langfuse()
client = OpenAI()

def answer_question(user_id: str, question: str):
    trace = lf.trace(
        name="rag-chat",
        user_id=user_id,
        input=question,
        metadata={"app": "yield-rag-bot"},
    )

    # 예: RAG 검색 단계
    search_obs = trace.span(
        name="vector-search",
        input={"query": question},
    )
    # ... 검색 코드 ...
    search_obs.update(
        output={"top_k_docs": ["doc1", "doc2"]},
    )

    # 실제 LLM 호출
    llm_obs = trace.generation(
        name="openai-chat",
        model="gpt-4.1-mini",
        input=question,
    )
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": question}],
    )
    answer = resp.choices[0].message.content
    llm_obs.update(output=answer)

    trace.update(output=answer)
    lf.flush()  # 버퍼링된 이벤트 전송

    return answer
#%%
user_id = "test_user"
question = "안녕 Langfuse"
answer = answer_question(user_id, question)
print(answer)
#%%
