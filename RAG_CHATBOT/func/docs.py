# 의존: faiss-cpu, numpy, langchain_community, langchain

# %%
import os
from pathlib import Path
from operator import itemgetter

import faiss
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

load_dotenv(override=True)

RETRIEVE_CHAIN_MODEL = os.getenv("RETRIEVE_CHAIN_MODEL")
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME")


# 호환성 임포트 유틸리티
def _try_import_class(module_paths, class_name):
    """여러 모듈 경로에서 클래스 import 시도"""
    for path in module_paths:
        try:
            module = __import__(path, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
    return None


# CacheBackedEmbeddings 호환 임포트 (버전별 경로)
_CacheBackedEmbeddings = _try_import_class(
    [
        "langchain.storage",
        "langchain_core.stores",
        "langchain.embeddings",
        "langchain.embeddings.cache",
    ],
    "CacheBackedEmbeddings",
)

# LocalFileStore 호환 임포트 (버전별 경로)
_LocalFileStore = _try_import_class(
    ["langchain_community.storage", "langchain.storage"],
    "LocalFileStore",
)


class NormalizingEmbeddings(Embeddings):
    def __init__(self, underlying: Embeddings):
        self.underlying = underlying

    @staticmethod
    def _normalize(text: str) -> str:
        q = str(text).replace("\r\n", "\n").replace("\r", "\n")
        q = "\n".join(line.rstrip() for line in q.split("\n")).strip()
        return q.casefold()

    def embed_query(self, text):
        return self.underlying.embed_query(self._normalize(text))

    def embed_documents(self, texts):
        normalized = [self._normalize(t) for t in texts]
        return self.underlying.embed_documents(normalized)


class MongoEmbeddingRetrievalChain:
    def __init__(self):
        self.embeddings = EMBEDDINGS_MODEL_NAME
        self.cache_dir = Path(".cache/embeddings")
        self.index_dir = Path(".cache/faiss_index")
        self.k = 10
        self.model_name = RETRIEVE_CHAIN_MODEL
        self.temperature = 0
        self.prompt = "teddynote/rag-prompt"
        self.cache_dir = Path(".cache/embeddings")
        self.index_dir = Path(".cache/faiss_index")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL")

    def create_prompt(self):
        return load_prompt(
            "/Users/daehwankim/Documents/langgraph-tutorial-main/RAG_CHATBOT/prompts/qa_prompt.yaml",
            encoding="utf-8",
        )

    def create_model(self):
        return ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def create_retriever(self, vectorstore):
        # Cosine Similarity 사용하여 검색을 수행하는 retriever를 생성합니다.
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return dense_retriever

    def create_chain(self):
        self.vectorstore = self.create_vectorstore_from_mongo()
        self.retriever = self.create_retriever(self.vectorstore)
        model = self.create_model()
        prompt = self.create_prompt()
        self.chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
                "chat_history": itemgetter("chat_history"),
            }
            | prompt
            | model
            | StrOutputParser()
        )
        return self

    def create_embedding(self):
        """임베딩 모델 생성 (캐싱 지원 시 자동 활성화)"""
        try:
            # 기본 임베딩 모델 생성
            underlying_embeddings = OpenAIEmbeddings(
                model=self.embeddings,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
            )

            # 캐싱 지원 여부 확인 및 적용
            if _CacheBackedEmbeddings and _LocalFileStore:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                store = _LocalFileStore(str(self.cache_dir))
                return _CacheBackedEmbeddings.from_bytes_store(
                    NormalizingEmbeddings(underlying_embeddings),
                    store,
                    namespace=self.embeddings,
                    key_encoder="sha256",
                )

            # 캐싱 미지원 시 기본 임베딩 반환
            return underlying_embeddings

        except Exception as e:
            # 예외 발생 시 기본 임베딩으로 폴백
            return OpenAIEmbeddings(
                model=self.embeddings,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
            )

    def create_vectorstore_from_mongo(self):
        records = (
            self._read_mongo()
        )  # [{ "text": str, "embedding": [float], "metadata": dict }, ...]
        vectors = np.array([r["embedding"] for r in records], dtype="float32")
        # 코사인 유사도 사용 시 사전 정규화 권장
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # 코사인: 정규화 + Inner Product
        index.add(vectors)

        docs = {
            str(i): Document(
                page_content=r["page_content"], metadata=r.get("metadata", {})
            )
            for i, r in enumerate(records)
        }
        id_map = {i: str(i) for i in range(len(records))}

        return FAISS(
            embedding_function=self.create_embedding(),  # 검색 시 쿼리 임베딩용
            index=index,
            docstore=InMemoryDocstore(docs),
            index_to_docstore_id=id_map,
        )

    def _read_mongo(self):
        with MongoClient("mongodb://localhost:27017/") as mongo_client:
            collection = mongo_client["document_vectorstore"]["embeddings"]
            projection = {"_id": 0, "page_content": 1, "metadata": 1, "embedding": 1}
            records = []
            append = records.append
            for document in collection.find({}, projection):
                embedding = document.get("embedding")
                if not embedding:
                    continue
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.astype("float32").tolist()
                else:
                    embedding = [float(value) for value in embedding]
                append(
                    {
                        "page_content": document.get("page_content", ""),
                        "metadata": document.get("metadata") or {},
                        "embedding": embedding,
                    }
                )
            return records
