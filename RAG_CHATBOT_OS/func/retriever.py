# %%
import os
from dotenv import load_dotenv
from pathlib import Path
from opensearchpy import OpenSearch
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# %%
# API KEY 정보로드
load_dotenv(override=True)
embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")

# OpenSearch 연결
opensearch_client = OpenSearch(
    hosts=[
        {
            "host": os.getenv("OPENSEARCH_HOST", "localhost"),
            "port": int(os.getenv("OPENSEARCH_PORT", "9200")),
        }
    ],
    http_auth=(
        os.getenv("OPENSEARCH_USER", "admin"),
        os.getenv("OPENSEARCH_PASSWORD", "admin"),
    ),
    use_ssl=os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true",
    verify_certs=False,
    ssl_show_warn=False,
)

OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "document_embeddings")
OPENSEARCH_EMBEDDING_FIELD = os.getenv("OPENSEARCH_EMBEDDING_FIELD", "embedding")

# %%
text_embedder = OpenAIEmbeddings(
    model=embeddings_model_name,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
)


# %%
# OpenSearch 기반 하이브리드 검색 Retriever
class UniversalOpenSearchRetriever:
    """OpenSearch에서 직접 하이브리드 검색하는 범용 Retriever"""

    def __init__(
        self, client, embedder, top_k=3, keyword_weight=0.3, semantic_weight=0.7
    ):
        self.client = client
        self.embedder = embedder
        self.top_k = top_k
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.index = OPENSEARCH_INDEX
        self.embedding_field = OPENSEARCH_EMBEDDING_FIELD

        # 인덱스 존재 확인
        if not self.client.indices.exists(index=self.index):
            print(f"경고: OpenSearch 인덱스 '{self.index}'가 존재하지 않습니다.")

        # k-NN 확장 설정
        self.use_expansion = os.getenv("USE_KNN_EXPANSION", "true").lower() == "true"
        self.expansion_k = int(
            os.getenv("KNN_EXPANSION_K", "5")
        )  # 각 문서당 확장할 이웃 수
        self.expansion_limit = int(
            os.getenv("KNN_EXPANSION_LIMIT", "10")
        )  # 최대 확장 문서 수

    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        q = str(text).replace("\r\n", "\n").replace("\r", "\n")
        q = "\n".join(line.rstrip() for line in q.split("\n")).strip()
        return q.casefold()

    def _normalize_scores(self, hits):
        """검색 결과의 score를 0-1로 정규화"""
        if not hits:
            return []

        scores = [hit["_score"] for hit in hits]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            for hit in hits:
                hit["_normalized_score"] = 1.0
        else:
            for hit in hits:
                normalized = (hit["_score"] - min_score) / (max_score - min_score)
                hit["_normalized_score"] = normalized

        return hits

    def _keyword_search(self, query: str, size: int, doc_type_filter=None):
        """BM25 키워드 검색"""
        query_body = {
            "multi_match": {
                "query": query,
                "fields": ["page_content"],
                "type": "best_fields",
            }
        }

        # doc_type 필터 추가
        if doc_type_filter:
            query_body = {
                "bool": {
                    "must": [query_body],
                    "filter": [{"term": {"metadata.doc_type": doc_type_filter}}],
                }
            }

        search_body = {
            "size": size,
            "query": query_body,
            "_source": ["page_content", "metadata"],
        }

        response = self.client.search(index=self.index, body=search_body)
        return response["hits"]["hits"]

    def _semantic_search(self, query_embedding: list, size: int, doc_type_filter=None):
        """kNN 벡터 검색"""
        knn_query = {
            self.embedding_field: {
                "vector": query_embedding,
                "k": size,
            }
        }

        # doc_type 필터 추가
        if doc_type_filter:
            knn_query[self.embedding_field]["filter"] = {
                "term": {"metadata.doc_type": doc_type_filter}
            }

        search_body = {
            "size": size,
            "query": {"knn": knn_query},
            "_source": ["page_content", "metadata"],
        }

        response = self.client.search(index=self.index, body=search_body)
        return response["hits"]["hits"]

    def search(
        self, query_text, top_k=None, doc_type_filter=None, search_mode="hybrid"
    ):
        """
        쿼리와 유사한 문서 검색

        Args:
            query_text: 검색 쿼리
            top_k: 반환할 문서 수
            doc_type_filter: 특정 문서 타입만 필터링 (예: "pptx", "excel", "pdf")
            search_mode: "hybrid" (기본), "semantic", "keyword"
        """
        k = top_k if top_k is not None else self.top_k
        fetch_size = k * 2  # 더 많이 가져와서 재순위화

        # 쿼리 정규화 및 임베딩
        normalized_query = self._normalize_text(query_text)

        try:
            if search_mode == "keyword":
                # 키워드 검색만
                keyword_hits = self._keyword_search(
                    normalized_query, k, doc_type_filter
                )
                results = self._format_results(keyword_hits, "keyword")

            elif search_mode == "semantic":
                # 시맨틱 검색만
                query_embedding = self.embedder.embed_query(normalized_query)
                semantic_hits = self._semantic_search(
                    query_embedding, k, doc_type_filter
                )
                results = self._format_results(semantic_hits, "semantic")

            else:  # hybrid
                # 하이브리드 검색
                query_embedding = self.embedder.embed_query(normalized_query)

                # 1. 키워드 검색
                keyword_hits = self._keyword_search(
                    normalized_query, fetch_size, doc_type_filter
                )
                keyword_hits = self._normalize_scores(keyword_hits)

                # 2. 시맨틱 검색
                semantic_hits = self._semantic_search(
                    query_embedding, fetch_size, doc_type_filter
                )
                semantic_hits = self._normalize_scores(semantic_hits)

                # 3. 결과 병합
                doc_scores = {}

                for hit in keyword_hits:
                    doc_id = hit["_id"]
                    score = hit["_normalized_score"] * self.keyword_weight
                    doc_scores[doc_id] = {"score": score, "hit": hit}

                for hit in semantic_hits:
                    doc_id = hit["_id"]
                    score = hit["_normalized_score"] * self.semantic_weight

                    if doc_id in doc_scores:
                        doc_scores[doc_id]["score"] += score
                    else:
                        doc_scores[doc_id] = {"score": score, "hit": hit}

                # 4. 정렬 및 상위 k개 선택
                sorted_docs = sorted(
                    doc_scores.items(),
                    key=lambda x: x[1]["score"],
                    reverse=True,
                )[:k]

                # 5. 결과 포맷
                results = []
                initial_doc_ids = set()  # 초기 검색 결과의 문서 ID 저장
                for doc_id, doc_data in sorted_docs:
                    hit = doc_data["hit"]
                    source = hit["_source"]
                    results.append(
                        {
                            "content": source.get("page_content", ""),
                            "metadata": source.get("metadata", {}),
                            "similarity": doc_data["score"],
                            "doc_id": doc_id,
                            "search_mode": "hybrid",
                        }
                    )
                    initial_doc_ids.add(doc_id)

                # 6. OpenSearch k-NN을 통한 문서 확장
                if self.use_expansion and initial_doc_ids:
                    expanded_docs = self._expand_documents_via_knn(
                        list(initial_doc_ids)
                    )

                    # 중복 제거 (이미 검색된 문서는 제외)
                    existing_doc_ids = {r["doc_id"] for r in results}
                    for doc_data in expanded_docs:
                        doc_id = doc_data["doc_id"]
                        if doc_id and doc_id not in existing_doc_ids:
                            results.append(doc_data)
                            existing_doc_ids.add(doc_id)
                            if len(existing_doc_ids) >= k + self.expansion_limit:
                                break

            return results

        except Exception as e:
            print(f"검색 오류: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _format_results(self, hits, search_mode):
        """검색 결과를 통일된 형식으로 변환"""
        results = []
        for hit in hits:
            source = hit["_source"]
            results.append(
                {
                    "content": source.get("page_content", ""),
                    "metadata": source.get("metadata", {}),
                    "similarity": hit["_score"],
                    "doc_id": hit["_id"],
                    "search_mode": search_mode,
                }
            )
        return results

    def _expand_documents_via_knn(self, doc_ids: list[str]) -> list[dict]:
        """OpenSearch k-NN을 사용하여 각 문서의 유사한 이웃 문서 찾기"""
        if not doc_ids:
            return []

        expanded_docs = []
        expanded_doc_ids = set()

        try:
            for doc_id in doc_ids[:5]:  # 최대 5개 문서에 대해서만 확장 (성능 고려)
                try:
                    # 해당 문서의 임베딩 가져오기
                    doc = self.client.get(
                        index=self.index,
                        id=doc_id,
                        _source=[self.embedding_field],
                    )
                    embedding = doc["_source"].get(self.embedding_field)

                    if not embedding:
                        continue

                    # k-NN 검색으로 유사한 문서 찾기
                    knn_query = {
                        self.embedding_field: {
                            "vector": embedding,
                            "k": self.expansion_k + 1,  # 자기 자신 제외
                        }
                    }

                    search_body = {
                        "size": self.expansion_k + 1,
                        "query": {"knn": knn_query},
                        "_source": ["page_content", "metadata"],
                    }

                    response = self.client.search(index=self.index, body=search_body)

                    # 중복 제거하며 추가
                    for hit in response["hits"]["hits"]:
                        neighbor_id = hit["_id"]
                        if (
                            neighbor_id != doc_id
                            and neighbor_id not in expanded_doc_ids
                        ):
                            source = hit["_source"]
                            expanded_docs.append(
                                {
                                    "content": source.get("page_content", ""),
                                    "metadata": source.get("metadata", {}),
                                    "similarity": hit["_score"],
                                    "doc_id": neighbor_id,
                                    "search_mode": "knn_expanded",
                                }
                            )
                            expanded_doc_ids.add(neighbor_id)

                            if len(expanded_docs) >= self.expansion_limit:
                                return expanded_docs

                except Exception as e:
                    # 개별 문서 처리 오류는 무시하고 계속 진행
                    continue

            return expanded_docs

        except Exception as e:
            print(f"k-NN 확장 검색 오류: {e}")
            import traceback

            traceback.print_exc()
            return []

    def invoke(self, query_text, doc_type_filter=None, search_mode="hybrid"):
        """LangChain 호환 인터페이스"""
        results = self.search(
            query_text, doc_type_filter=doc_type_filter, search_mode=search_mode
        )
        return [
            Document(
                page_content=r["content"],
                metadata={
                    **r["metadata"],
                    "similarity_score": r["similarity"],
                    "search_mode": r["search_mode"],
                },
            )
            for r in results
        ]


# %%
# Retriever 생성 (OpenSearch 기반)
# top_k는 환경변수에서 읽기 (기본값 5)
retriever_top_k = int(os.getenv("RETRIEVE_K_PER_TYPE", "5"))
retriever = UniversalOpenSearchRetriever(
    client=opensearch_client,
    embedder=text_embedder,
    top_k=retriever_top_k,
    keyword_weight=0.3,  # 키워드 30%
    semantic_weight=0.7,  # 시맨틱 70%
)

# 검색 테스트
query = """
word2vec이 뭐야
"""

# 하이브리드 검색 (기본)
print(f"\n[하이브리드 검색] 쿼리: '{query.strip()}'")
results = retriever.search(query, top_k=5, search_mode="hybrid")
for i, r in enumerate(results, 1):
    print(f"\n{i}. [{r['metadata']['doc_type']}] {Path(r['metadata']['source']).name}")
    print(f"   유사도: {r['similarity']:.4f}")
    print(f"   내용: {r['content'][:150]}...")

# 시맨틱 검색만
print(f"\n[시맨틱 검색만] 쿼리: '{query.strip()}'")
semantic_results = retriever.search(query, top_k=5, search_mode="semantic")
for i, r in enumerate(semantic_results, 1):
    print(f"\n{i}. [{r['metadata']['doc_type']}] {Path(r['metadata']['source']).name}")
    print(f"   유사도: {r['similarity']:.4f}")

# 특정 문서 타입만 검색
print(f"\n[PPTX만 검색] 쿼리: '{query.strip()}'")
pptx_results = retriever.search(query, top_k=3, doc_type_filter="pptx")
for i, r in enumerate(pptx_results, 1):
    print(f"\n{i}. 슬라이드 {r['metadata'].get('page_number')}")
    print(f"   유사도: {r['similarity']:.4f}")

# %%
