# %%
import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


def normalize_text(text: str) -> str:
    q = str(text).replace("\r\n", "\n").replace("\r", "\n")
    q = "\n".join(line.rstrip() for line in q.split("\n")).strip()
    return q.casefold()


load_dotenv(override=True)


client = OpenAI(
    # model=os.getenv("EMBEDDINGS_MODEL_NAME"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)


def get_embedding(text):
    res = client.embeddings.create(model="qwen/qwen3-embedding-8b", input=text)
    emb = np.array(res.data[0].embedding, dtype=np.float32)
    return normalize(emb.reshape(1, -1))[0]


embedder = OpenAIEmbeddings(
    model=os.getenv("EMBEDDINGS_MODEL_NAME"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
)

text_a = "Isotropic etch는 등방성 식각으로 모든 방향에 동일한 식률로 etching됨. 주로 원형태로 식각됨"
text_b = "SC2 BRG의 불량 MAP 형태는 Edge에 동그랗게 EASY(W) & DIST(D) & Shot 성(Overlay 기인)으로 나타난다."
text_c = "SC2 BRG의 불량 MAP은 어떤 형태로 나타나?"

a_norm = normalize_text(text_a)
b_norm = normalize_text(text_b)

vec_a = np.array(get_embedding(a_norm), dtype=np.float32)
vec_b = np.array(get_embedding(b_norm), dtype=np.float32)

# L2 정규화 (코사인 유사도 계산용)
vec_a = vec_a / (np.linalg.norm(vec_a) + 1e-12)
vec_b = vec_b / (np.linalg.norm(vec_b) + 1e-12)

cosine_sim = float(np.dot(vec_a, vec_b))
print("cosine_similarity:", cosine_sim)


# %%
from openai import OpenAI
import numpy as np
import re
from sklearn.preprocessing import normalize

client = OpenAI(
    # model=os.getenv("EMBEDDINGS_MODEL_NAME"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)

# --------------------------------------------
# 1️⃣ 데이터 정의
# --------------------------------------------
text_a = "Isotropic etch는 등방성 식각으로 모든 방향에 동일한 식률로 etching됨. 주로 원형태로 식각됨"
text_b = "SC2 BRG의 불량 MAP 형태는 Edge에 동그랗게 EASY(W) & DIST(D) & Shot 성(Overlay 기인)으로 나타난다."
query = "SC2 BRG의 불량 MAP 형태는 뭐야?"
query_rewrite = "SC2 BRG 불량 MAP 형태 설명"


# --------------------------------------------
# 2️⃣ 도메인 전처리 함수
# --------------------------------------------
def preprocess_domain_terms(text: str):
    text = re.sub(r"([A-Z]+)(\d+)", r"\1 \2", text)  # SC2 → SC 2
    text = re.sub(r"([A-Z]{2,})", r"\1 ", text)  # BRG → BRG
    return text.strip()


# --------------------------------------------
# 3️⃣ 임베딩 요청 함수
# --------------------------------------------
def get_embedding(text):
    res = client.embeddings.create(model="qwen/qwen3-embedding-8b", input=text)
    emb = np.array(res.data[0].embedding, dtype=np.float32)
    return normalize(emb.reshape(1, -1))[0]


# --------------------------------------------
# 4️⃣ 임베딩 계산
# --------------------------------------------
emb_query = get_embedding(query)
emb_query_re = get_embedding(query_rewrite)
emb_query_proc = get_embedding(preprocess_domain_terms(query))

emb_a = get_embedding(text_a)
emb_b = get_embedding(text_b)
emb_b_proc = get_embedding(preprocess_domain_terms(text_b))


# --------------------------------------------
# 5️⃣ 유사도 비교
# --------------------------------------------
def cos(a, b):
    return float(np.dot(a, b))


print("\n=== Cosine Similarity 결과 ===")
print(f"원 query vs text_a : {cos(emb_query, emb_a):.3f}")
print(f"원 query vs text_b : {cos(emb_query, emb_b):.3f}")
print(f"rewriting query vs text_b : {cos(emb_query_re, emb_b):.3f}")
print(f"전처리(query,text_b) vs text_b_proc : {cos(emb_query_proc, emb_b_proc):.3f}")

# %%


# pip install openai langchain-openai scikit-learn python-dotenv

# pip install -U openai langchain-openai scikit-learn python-dotenv

import os, numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import normalize as l2norm
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = "qwen/qwen3-embedding-8b"  # 양쪽 동일


def norm_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(line.rstrip() for line in t.split("\n")).strip()
    return t.replace("\n", " ")


def add_prefix(t: str, mode: str):
    return f"query: {t}" if mode == "query" else f"passage: {t}"


# ---- A) 직접 호출 ----
oclient = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def embed_direct(texts, mode="passage"):
    texts = [add_prefix(norm_text(x), mode) for x in texts]
    r = oclient.embeddings.create(model=MODEL, input=texts)
    print("[direct] model used:", r.model)  # 실제 사용 모델 확인
    vecs = np.array([d.embedding for d in r.data], dtype=np.float32)
    return l2norm(vecs)


# ---- B) LangChain을 '같은 클라이언트'로 강제 ----
# 핵심: LangChain 내부 client가 사용하는 SDK 객체를 '같은' oclient.embeddings로 고정
emb_lc = OpenAIEmbeddings(
    model=MODEL,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    check_embedding_ctx_length=False,  # ★ 핵심: 토큰화 비활성화!
)


def embed_langchain_docs(texts):
    texts = [add_prefix(norm_text(x), "passage") for x in texts]
    # 로깅: LangChain이 진짜 어디로 치는지 확인
    r = emb_lc.client.create(model=MODEL, input=[texts[0]])
    print("[langchain] model used (probe):", r.model)
    vecs = emb_lc.embed_documents(texts)
    return l2norm(np.array(vecs, dtype=np.float32))


def embed_langchain_query(text):
    text = add_prefix(norm_text(text), "query")
    # 로깅: 쿼리 경로도 동일 모델 확인
    r = emb_lc.client.create(model=MODEL, input=[text])
    print("[langchain] model used (probe q):", r.model)
    vec = emb_lc.embed_query(text)
    return l2norm(np.array([vec], dtype=np.float32))[0]


# ---- 테스트 ----
text_b = "SC2 BRG의 불량 MAP 형태는 Edge에 동그랗게 EASY(W) & DIST(D) & Shot 성(Overlay 기인)으로 나타난다."
text_c = "SC2 BRG의 불량 MAP은 어떤 형태로 나타나?"

v_b_direct = embed_direct([text_b], "passage")[0]
v_c_direct = embed_direct([text_c], "query")[0]
print("cosine (direct):", float(np.dot(v_b_direct, v_c_direct)))

v_b_lc = embed_langchain_docs([text_b])[0]
v_c_lc = embed_langchain_query(text_c)
print("cosine (langchain):", float(np.dot(v_b_lc, v_c_lc)))

# %%
