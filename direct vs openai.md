# OpenAI Embeddings: 직접 호출 vs LangChain 성능 차이 분석

## 📋 문제 상황

동일한 텍스트를 임베딩할 때 두 방식의 코사인 유사도가 크게 차이남:

- **직접 호출**: 0.86
- **LangChain**: 0.50

```python
text_b = "SC2 BRG의 불량 MAP 형태는 Edge에 동그랗게 EASY(W) & DIST(D) & Shot 성(Overlay 기인)으로 나타난다."
text_c = "SC2 BRG의 불량 MAP은 어떤 형태로 나타나?"
```

---

## 🔍 원인 분석

### 1. API에 전송되는 데이터 형식이 다름

#### 직접 호출 (OpenAI SDK)

```python
# 전송되는 데이터: 문자열
'passage: SC2 BRG의 불량 MAP 형태는 Edge에 동그랗게 EASY(W) & DIST(D) & Shot 성(Overlay 기인)으로 나타난다.'
```

#### LangChain (기본 설정)

```python
# 전송되는 데이터: 토큰 ID 배열
[6519, 425, 25, 7683, 17, 19333, 38, 21028, 5251, 28857, ...]
```

### 2. 근본 원인: `check_embedding_ctx_length` 파라미터

LangChain의 `OpenAIEmbeddings`는 기본적으로 `check_embedding_ctx_length=True`로 설정되어 있어:

1. 텍스트를 `tiktoken`으로 **토큰화**
2. 토큰 길이를 체크하여 컨텍스트 길이 제한 관리
3. 토큰 ID 배열을 그대로 API에 전송
4. **모델이 문자열과 토큰 배열을 다르게 처리** → 임베딩 결과가 달라짐

---

## ✅ 해결 방법

### Before (문제 있는 코드)

```python
from langchain_openai import OpenAIEmbeddings

emb_lc = OpenAIEmbeddings(
    model="qwen/qwen3-embedding-8b",
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    # check_embedding_ctx_length 기본값: True
)

# 결과: 0.50 (낮은 유사도)
```

### After (수정된 코드)

```python
from langchain_openai import OpenAIEmbeddings

emb_lc = OpenAIEmbeddings(
    model="qwen/qwen3-embedding-8b",
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    check_embedding_ctx_length=False,  # ← 핵심!
)

# 결과: 0.86 (직접 호출과 동일)
```

---

## 📊 결과 비교

| 방식             | 전송 형태    | Cosine 유사도    | 비고    |
| ---------------- | ------------ | ---------------- | ------- |
| 직접 호출        | 문자열       | **0.8364** | ✅ 정상 |
| LangChain (기본) | 토큰 ID 배열 | **0.5649** | ❌ 문제 |
| LangChain (수정) | 문자열       | **0.8364** | ✅ 해결 |

---

## 🔧 완전한 코드 예시

```python
import os
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import normalize as l2norm
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

API_KEY  = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL    = "qwen/qwen3-embedding-8b"

def norm_text(t: str) -> str:
    t = t.replace("\r\n","\n").replace("\r","\n")
    t = "\n".join(line.rstrip() for line in t.split("\n")).strip()
    return t.replace("\n", " ")

def add_prefix(t: str, mode: str):
    return f"query: {t}" if mode == "query" else f"passage: {t}"

# ---- A) 직접 호출 ----
oclient = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def embed_direct(texts, mode="passage"):
    texts = [add_prefix(norm_text(x), mode) for x in texts]
    r = oclient.embeddings.create(model=MODEL, input=texts)
    vecs = np.array([d.embedding for d in r.data], dtype=np.float32)
    return l2norm(vecs)

# ---- B) LangChain - 수정된 버전 ----
emb_lc = OpenAIEmbeddings(
    model=MODEL,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    check_embedding_ctx_length=False,  # ★ 핵심!
)

def embed_langchain_docs(texts):
    texts = [add_prefix(norm_text(x), "passage") for x in texts]
    vecs = emb_lc.embed_documents(texts)
    return l2norm(np.array(vecs, dtype=np.float32))

def embed_langchain_query(text):
    text = add_prefix(norm_text(text), "query")
    vec = emb_lc.embed_query(text)
    return l2norm(np.array([vec], dtype=np.float32))[0]

# ---- 테스트 ----
text_b = "SC2 BRG의 불량 MAP 형태는 Edge에 동그랗게 EASY(W) & DIST(D) & Shot 성(Overlay 기인)으로 나타난다."
text_c = "SC2 BRG의 불량 MAP은 어떤 형태로 나타나?"

v_b_direct = embed_direct([text_b], "passage")[0]
v_c_direct = embed_direct([text_c], "query")[0]
print(f"cosine (직접):    {float(np.dot(v_b_direct, v_c_direct)):.4f}")

v_b_lc = embed_langchain_docs([text_b])[0]
v_c_lc = embed_langchain_query(text_c)
print(f"cosine (LangChain): {float(np.dot(v_b_lc, v_c_lc)):.4f}")

# 출력:
# cosine (직접):    0.8364
# cosine (LangChain): 0.8364
```

---

## 💡 `check_embedding_ctx_length` 파라미터 이해

### `True` (기본값)

- **장점**:
  - 긴 텍스트 자동 처리 (청킹 & 가중 평균)
  - 컨텍스트 길이 제한 자동 관리
- **단점**:
  - 토큰화로 인해 임베딩 품질이 달라질 수 있음
  - 예상치 못한 결과 발생 가능

### `False` (권장)

- **장점**:
  - 텍스트를 있는 그대로 전송
  - 직접 호출과 동일한 결과
  - 예측 가능한 동작
- **단점**:
  - 매우 긴 텍스트는 수동으로 처리 필요

---

## 🎯 결론

### 핵심 교훈

1. LangChain의 **기본 설정이 항상 최선은 아님**
2. 임베딩 품질이 중요한 경우 `check_embedding_ctx_length=False` 사용 권장
3. 프로덕션 환경에서는 **직접 호출과 비교 테스트** 필수

### 적용 권장 사항

```python
# RAG, 검색 등 임베딩 품질이 중요한 경우
emb = OpenAIEmbeddings(
    model="your-model",
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    check_embedding_ctx_length=False,  # 권장!
)
```

---

## 📚 참고 자료

### LangChain 내부 동작

```python
# embed_query는 내부적으로 embed_documents를 호출
def embed_query(self, text: str, **kwargs: Any) -> list[float]:
    return self.embed_documents([text], **kwargs)[0]

# embed_documents는 check_embedding_ctx_length에 따라 분기
def embed_documents(self, texts: list[str], ...):
    if not self.check_embedding_ctx_length:
        # 텍스트를 그대로 전송
        response = self.client.create(input=texts, ...)
    else:
        # 토큰화 후 전송
        return self._get_len_safe_embeddings(texts, ...)
```

### 디버깅 팁

API에 실제로 전송되는 데이터를 확인하려면:

```python
# client.create 메서드를 래핑하여 로깅
original_create = emb_lc.client.create

def debug_create(*args, **kwargs):
    print("Sending to API:", kwargs.get('input'))
    return original_create(*args, **kwargs)

emb_lc.client.create = debug_create
```

---

**작성일**: 2025-11-03
**분석 도구**: Python, OpenAI SDK, LangChain, NumPy, scikit-learn
