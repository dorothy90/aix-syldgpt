# %%
from pptx import Presentation
from openpyxl import load_workbook
import os
from dotenv import load_dotenv
import re
from pathlib import Path
import base64
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pymongo import MongoClient
import numpy as np
import faiss

# %%
# API KEY 정보로드
load_dotenv(override=True)
embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")
# MongoDB 연결
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["document_vectorstore"]
collection = db["embeddings"]

# %%
text_embedder = OpenAIEmbeddings(
    model=embeddings_model_name,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
)


# %%
# 5. FAISS 기반 벡터 검색 Retriever
class UniversalFAISSRetriever:
    """MongoDB에서 임베딩을 가져와 FAISS로 검색하는 범용 Retriever"""

    def __init__(self, collection, embedder, top_k=3):
        self.collection = collection
        self.embedder = embedder
        self.top_k = top_k
        self.index = None
        self.doc_metadata = []
        self.embedding_dim = None

        # 초기화 시 MongoDB에서 데이터 로드 및 FAISS 인덱스 생성
        self._build_faiss_index()

    def _build_faiss_index(self):
        """MongoDB에서 embedding을 가져와 FAISS 인덱스 생성"""
        print("MongoDB에서 embedding 데이터 로드 중...")

        # MongoDB에서 모든 문서 가져오기
        all_docs = list(self.collection.find())

        if not all_docs:
            print("경고: MongoDB에 문서가 없습니다.")
            return

        # embedding 벡터와 메타데이터 분리
        embeddings = []
        self.doc_metadata = []

        for doc in all_docs:
            embeddings.append(doc["embedding"])
            self.doc_metadata.append(
                {
                    "doc_id": str(doc["_id"]),
                    "content": doc["page_content"],
                    "metadata": doc["metadata"],
                }
            )

        # numpy 배열로 변환
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.embedding_dim = embeddings_array.shape[1]

        # FAISS 인덱스 생성 (코사인 유사도를 위해 정규화 후 내적 사용)
        # L2 정규화
        faiss.normalize_L2(embeddings_array)

        # Inner Product 인덱스 생성 (정규화된 벡터의 내적 = 코사인 유사도)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_array)

        print(
            f"✅ FAISS 인덱스 생성 완료: {len(self.doc_metadata)}개 문서, 차원: {self.embedding_dim}"
        )

    def search(self, query_text, top_k=None, doc_type_filter=None):
        """
        쿼리와 유사한 문서 검색

        Args:
            query_text: 검색 쿼리
            top_k: 반환할 문서 수
            doc_type_filter: 특정 문서 타입만 필터링 (예: "pptx", "excel", "pdf")
        """
        if self.index is None or len(self.doc_metadata) == 0:
            print("경고: FAISS 인덱스가 비어있습니다.")
            return []

        k = top_k if top_k is not None else self.top_k

        # 입력 텍스트 정규화 (형식을 최대한 보존: 줄바꿈 유지)
        # - CRLF/CR -> LF로 통일
        # - 각 라인 끝 공백 제거, 전체 앞뒤 공백 제거
        q = str(query_text).replace("\r\n", "\n").replace("\r", "\n")
        q = "\n".join(line.rstrip() for line in q.split("\n")).strip()
        normalized_query = q.casefold()

        # 쿼리 임베딩 생성
        query_embedding = self.embedder.embed_query(normalized_query)
        query_vector = np.array([query_embedding], dtype=np.float32)

        # L2 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(query_vector)

        # doc_type_filter가 있는 경우 필터링된 인덱스로 검색
        if doc_type_filter:
            # 필터링된 메타데이터와 인덱스 생성
            filtered_indices = [
                i
                for i, meta in enumerate(self.doc_metadata)
                if meta["metadata"].get("doc_type") == doc_type_filter
            ]

            if not filtered_indices:
                return []

            # 필터링된 임베딩으로 임시 인덱스 생성
            filtered_embeddings = []
            for idx in filtered_indices:
                # 원본 인덱스에서 벡터 재구성
                vec = self.index.reconstruct(int(idx))
                filtered_embeddings.append(vec)

            filtered_embeddings_array = np.array(filtered_embeddings, dtype=np.float32)
            temp_index = faiss.IndexFlatIP(self.embedding_dim)
            temp_index.add(filtered_embeddings_array)

            # 검색 수행
            k_search = min(k, len(filtered_indices))
            similarities, indices = temp_index.search(query_vector, k_search)

            # 원본 인덱스로 매핑
            original_indices = [filtered_indices[idx] for idx in indices[0]]
            similarities = similarities[0]
        else:
            # 전체 인덱스에서 검색
            k_search = min(k, len(self.doc_metadata))
            similarities, indices = self.index.search(query_vector, k_search)
            original_indices = indices[0]
            similarities = similarities[0]

        # 결과 구성
        results = []
        for idx, similarity in zip(original_indices, similarities):
            meta = self.doc_metadata[idx]
            results.append(
                {
                    "content": meta["content"],
                    "metadata": meta["metadata"],
                    "similarity": float(similarity),
                    "doc_id": meta["doc_id"],
                }
            )

        return results

    def invoke(self, query_text, doc_type_filter=None):
        """LangChain 호환 인터페이스"""
        results = self.search(query_text, doc_type_filter=doc_type_filter)
        return [
            Document(
                page_content=r["content"],
                metadata={**r["metadata"], "similarity_score": r["similarity"]},
            )
            for r in results
        ]

    def rebuild_index(self):
        """인덱스를 다시 빌드 (MongoDB 데이터가 업데이트된 경우)"""
        self._build_faiss_index()


# %%
# Retriever 생성 (FAISS 기반)
retriever = UniversalFAISSRetriever(
    collection=collection, embedder=text_embedder, top_k=5
)

# 검색 테스트
query = """
word2vec이 뭐야
"""

# 전체 문서에서 검색
print(f"\n[전체 문서 검색] 쿼리: '{query}'")
results = retriever.search(query, top_k=5)
for i, r in enumerate(results, 1):
    print(f"\n{i}. [{r['metadata']['doc_type']}] {Path(r['metadata']['source']).name}")
    print(f"   유사도: {r['similarity']:.4f}")
    print(f"   내용: {r['content'][:150]}...")

# # PPTX만 검색
# print(f"\n[PPTX만 검색] 쿼리: '{query}'")
# pptx_results = retriever.search(query, top_k=3, doc_type_filter="pptx")
# for i, r in enumerate(pptx_results, 1):
#     print(f"\n{i}. 슬라이드 {r['metadata'].get('page_number')}")
#     print(f"   유사도: {r['similarity']:.4f}")

# %%
