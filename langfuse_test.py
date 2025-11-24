#%%
from langfuse import observe, get_client
from openai import OpenAI

client = OpenAI()  # 기존에 쓰던 OpenAI 클라이언트
langfuse = get_client()    # Langfuse 클라이언트 (전역으로 하나 두고 사용)
# Cell 2: Langfuse로 트레이싱되는 함수
@observe(name="rag-chat", as_type="chain")  # 함수 하나가 하나의 trace + root span이 됨
def answer_question(user_id: str, question: str):
    # ---- (1) RAG 검색 단계 예시 span ----
    with langfuse.start_as_current_observation(
        as_type="retriever",
        name="vector-search",
        input={"query": question, "user_id": user_id},
    ) as search_span:
        # TODO: 실제 벡터 검색 코드 자리
        # search_results = my_vector_search(question)
        search_results = ["doc1", "doc2"]  # 데모용
        search_span.update(output={"top_k_docs": search_results})

    # ---- (2) LLM 호출 generation ----
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="openai-chat",
        model="gpt-4.1-mini",
        input=question,
    ) as gen:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": question}],
        )
        answer = resp.choices[0].message.content
        gen.update(output=answer)

    # 필요하면 여기서 user_id 같은 걸 trace attribute로도 올릴 수 있음
    # (조금 더 고급 옵션)

    return answer

#%%
# Cell 3: 실제 호출 + flush
user_id = "test_user"
question = "안녕 Langfuse"

answer = answer_question(user_id=user_id, question=question)
print(answer)

# 짧게 한 번만 돌리는 스크립트면 flush
langfuse.flush()

# %%
