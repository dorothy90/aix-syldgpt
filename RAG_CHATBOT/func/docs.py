# 의존: faiss-cpu, numpy, langchain_community, langchain
import faiss, numpy as np
from langchain_community.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from pathlib import Path
import os
from langchain import hub
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.embeddings import Embeddings


class NormalizingEmbeddings(Embeddings):
    def __init__(self, underlying: Embeddings):
        self.underlying = underlying

    @staticmethod
    def _normalize(text: str) -> str:
        q = str(text).replace("\r\n", "\n").replace("\r", "\n")
        q = "\n".join(line.rstrip() for line in q.split("\n")).strip()
        return q

    def embed_query(self, text):
        return self.underlying.embed_query(self._normalize(text))

    def embed_documents(self, texts):
        normalized = [self._normalize(t) for t in texts]
        return self.underlying.embed_documents(normalized)


class MongoEmbeddingRetrievalChain:
    def __init__(self):
        self.embeddings = "qwen/qwen3-embedding-8b"
        self.cache_dir = Path(".cache/embeddings")
        self.index_dir = Path(".cache/faiss_index")
        self.k = 15
        self.model_name = "gpt-4.1"
        self.temperature = 0
        self.prompt = "teddynote/rag-prompt"
        self.cache_dir = Path(".cache/embeddings")
        self.index_dir = Path(".cache/faiss_index")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL")

    def create_prompt(self):
        return hub.pull(self.prompt)

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
            {"question": itemgetter("question"), "context": itemgetter("context")}
            | prompt
            | model
            | StrOutputParser()
        )
        return self

    def create_embedding(self):
        try:
            # 캐시 디렉토리 생성
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # 기본 임베딩 모델 생성 (OpenRouter 설정 포함)
            underlying_embeddings = OpenAIEmbeddings(
                model=self.embeddings,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
            )

            # 파일 기반 캐시 스토어 생성
            store = LocalFileStore(str(self.cache_dir))

            # 캐시 기반 임베딩 생성 (SHA-256 사용으로 보안 강화)
            cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
                NormalizingEmbeddings(underlying_embeddings),
                store,
                namespace=self.embeddings,
                key_encoder="sha256",
            )

            return cached_embeddings
        except Exception as e:
            print(f"Warning: Failed to create cached embeddings: {e}")
            print("Falling back to basic OpenAI embeddings without caching")
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
