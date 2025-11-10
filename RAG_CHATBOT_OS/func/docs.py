# 의존: langchain_community, langchain, opensearch-py

# %%
import os
from operator import itemgetter

from dotenv import load_dotenv
from opensearchpy import OpenSearch

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv(override=True)

RETRIEVE_CHAIN_MODEL = os.getenv("RETRIEVE_CHAIN_MODEL")
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME")


class OpenSearchEmbeddingRetrievalChain:
    def __init__(self):
        self.embeddings = EMBEDDINGS_MODEL_NAME
        self.k = 10
        self.model_name = RETRIEVE_CHAIN_MODEL
        self.temperature = 0
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL")

        # OpenSearch 설정
        self.opensearch_host = os.getenv("OPENSEARCH_HOST", "localhost")
        self.opensearch_port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        self.opensearch_user = os.getenv("OPENSEARCH_USER", "admin")
        self.opensearch_password = os.getenv("OPENSEARCH_PASSWORD", "admin")
        self.opensearch_index = os.getenv("OPENSEARCH_INDEX", "document_embeddings")
        self.opensearch_use_ssl = (
            os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
        )
        self.opensearch_embedding_field = os.getenv(
            "OPENSEARCH_EMBEDDING_FIELD", "embedding"
        )

        # 하이브리드 검색 가중치 (키워드: 시맨틱 = 3:7)
        self.keyword_weight = 0.3
        self.semantic_weight = 0.7

        # doc_type별 검색 결과 개수 (환경변수에서 읽기, 기본값 30)
        self.k_per_type = int(os.getenv("RETRIEVE_K_PER_TYPE", "30"))

        # k-NN 확장 설정
        self.use_expansion = os.getenv("USE_KNN_EXPANSION", "true").lower() == "true"
        self.expansion_k = int(
            os.getenv("KNN_EXPANSION_K", "5")
        )  # 각 문서당 확장할 이웃 수
        self.expansion_limit = int(
            os.getenv("KNN_EXPANSION_LIMIT", "10")
        )  # 최대 확장 문서 수

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

    def create_embedding(self):
        """임베딩 모델 생성"""
        return OpenAIEmbeddings(
            model=self.embeddings,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
        )

    def _get_opensearch_client(self):
        """OpenSearch 클라이언트 생성"""
        return OpenSearch(
            hosts=[{"host": self.opensearch_host, "port": self.opensearch_port}],
            http_auth=(self.opensearch_user, self.opensearch_password),
            use_ssl=self.opensearch_use_ssl,
            verify_certs=False,
            ssl_show_warn=False,
        )

    def _normalize_scores(self, hits):
        """검색 결과의 score를 0-1로 정규화"""
        if not hits:
            return []

        scores = [hit["_score"] for hit in hits]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # 모든 스코어가 동일한 경우
            for hit in hits:
                hit["_normalized_score"] = 1.0
        else:
            # Min-Max 정규화
            for hit in hits:
                normalized = (hit["_score"] - min_score) / (max_score - min_score)
                hit["_normalized_score"] = normalized

        return hits

    def _keyword_search(
        self, client, query: str, size: int, doc_type_filter: str = None
    ):
        """BM25 키워드 검색 - 부분 매칭 지원"""

        # 쿼리의 주요 키워드 추출 (예: "WLBL Test" → ["WLBL", "Test"])
        keywords = query.split()

        # 기본 쿼리 구성 - 여러 방식으로 검색
        base_query_clause = {
            "bool": {
                "should": [
                    # 1. 정확한 전체 쿼리 매칭
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["page_content"],
                            "type": "best_fields",
                            "boost": 2.0,  # 정확한 매칭에 높은 가중치
                        }
                    },
                    # 2. 각 키워드별로 부분 매칭 (wildcard)
                    *[
                        {
                            "wildcard": {
                                "page_content": {"value": f"*{keyword}*", "boost": 1.5}
                            }
                        }
                        for keyword in keywords
                        if len(keyword) >= 2  # 2글자 이상만
                    ],
                    # 3. 각 키워드로 일반 검색
                    *[
                        {"match": {"page_content": {"query": keyword, "boost": 1.0}}}
                        for keyword in keywords
                    ],
                ],
                "minimum_should_match": 1,  # 최소 하나는 매칭되어야 함
            }
        }

        # doc_type 필터 추가
        if doc_type_filter is not None:
            if doc_type_filter == "parquet":
                # parquet인 경우
                query_clause = {
                    "bool": {
                        "must": [
                            base_query_clause,
                            {"term": {"metadata.doc_type": "parquet"}},
                        ]
                    }
                }
            else:
                # parquet이 아닌 경우
                query_clause = {
                    "bool": {
                        "must": [
                            base_query_clause,
                            {
                                "bool": {
                                    "must_not": {
                                        "term": {"metadata.doc_type": "parquet"}
                                    }
                                }
                            },
                        ]
                    }
                }
        else:
            query_clause = base_query_clause

        search_body = {
            "size": size,
            "query": query_clause,
            "_source": ["page_content", "metadata"],
        }

        response = client.search(index=self.opensearch_index, body=search_body)
        return response["hits"]["hits"]

    def _semantic_search(
        self, client, query_embedding: list, size: int, doc_type_filter: str = None
    ):
        """kNN 벡터 검색"""
        knn_clause = {
            self.opensearch_embedding_field: {
                "vector": query_embedding,
                "k": size,
            }
        }

        # doc_type 필터 추가
        filter_clause = None
        if doc_type_filter is not None:
            if doc_type_filter == "parquet":
                # parquet인 경우
                filter_clause = {"term": {"metadata.doc_type": "parquet"}}
            else:
                # parquet이 아닌 경우
                filter_clause = {
                    "bool": {"must_not": {"term": {"metadata.doc_type": "parquet"}}}
                }

        search_body = {
            "size": size,
            "query": {"knn": knn_clause},
            "_source": ["page_content", "metadata"],
        }

        # 필터가 있는 경우 추가
        if filter_clause:
            search_body["query"]["knn"][self.opensearch_embedding_field][
                "filter"
            ] = filter_clause

        response = client.search(index=self.opensearch_index, body=search_body)
        return response["hits"]["hits"]

    def _expand_documents_via_knn(self, client, doc_ids: list[str]) -> list[Document]:
        """OpenSearch k-NN을 사용하여 각 문서의 유사한 이웃 문서 찾기"""
        if not doc_ids:
            return []

        expanded_docs = []
        expanded_doc_ids = set()

        try:
            for doc_id in doc_ids[:5]:  # 최대 5개 문서에 대해서만 확장 (성능 고려)
                try:
                    # 해당 문서의 임베딩 가져오기
                    doc = client.get(
                        index=self.opensearch_index,
                        id=doc_id,
                        _source=[self.opensearch_embedding_field],
                    )
                    embedding = doc["_source"].get(self.opensearch_embedding_field)

                    if not embedding:
                        continue

                    # k-NN 검색으로 유사한 문서 찾기
                    knn_query = {
                        self.opensearch_embedding_field: {
                            "vector": embedding,
                            "k": self.expansion_k + 1,  # 자기 자신 제외
                        }
                    }

                    search_body = {
                        "size": self.expansion_k + 1,
                        "query": {"knn": knn_query},
                        "_source": ["page_content", "metadata"],
                    }

                    response = client.search(
                        index=self.opensearch_index, body=search_body
                    )

                    # 중복 제거하며 추가
                    for hit in response["hits"]["hits"]:
                        neighbor_id = hit["_id"]
                        if (
                            neighbor_id != doc_id
                            and neighbor_id not in expanded_doc_ids
                        ):
                            source = hit["_source"]
                            expanded_docs.append(
                                Document(
                                    page_content=source.get("page_content", ""),
                                    metadata={
                                        **source.get("metadata", {}),
                                        "doc_id": neighbor_id,
                                        "knn_expanded": True,
                                    },
                                )
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

    def search_similar_documents(self, query: str) -> list[Document]:
        """하이브리드 검색 수행 (키워드 30% + 시맨틱 70%)
        doc_type이 parquet인 것과 아닌 것을 각각 k_per_type개씩 검색
        OpenSearch k-NN을 통해 추가 문서 확장"""
        # 쿼리를 임베딩으로 변환
        embedding_model = self.create_embedding()
        query_embedding = embedding_model.embed_query(query)

        client = self._get_opensearch_client()

        try:
            # 각 doc_type별로 k_per_type개씩 가져오기 위해 더 많은 결과를 가져와서 재순위화
            fetch_size = self.k_per_type * 2

            all_documents = []
            initial_doc_ids = set()  # 초기 검색 결과의 문서 ID 저장

            # doc_type별로 검색 수행
            for doc_type_filter in ["parquet", "non_parquet"]:
                # 1. 키워드 검색 (BM25)
                keyword_hits = self._keyword_search(
                    client, query, fetch_size, doc_type_filter
                )
                keyword_hits = self._normalize_scores(keyword_hits)

                # 2. 시맨틱 검색 (kNN)
                semantic_hits = self._semantic_search(
                    client, query_embedding, fetch_size, doc_type_filter
                )
                semantic_hits = self._normalize_scores(semantic_hits)

                # 3. 결과 병합 및 점수 계산
                doc_scores = {}  # {doc_id: {"score": float, "hit": dict}}

                # 키워드 검색 결과 추가 (30% 가중치)
                for hit in keyword_hits:
                    doc_id = hit["_id"]
                    score = hit["_normalized_score"] * self.keyword_weight
                    doc_scores[doc_id] = {"score": score, "hit": hit}
                    initial_doc_ids.add(doc_id)

                # 시맨틱 검색 결과 추가 (70% 가중치)
                for hit in semantic_hits:
                    doc_id = hit["_id"]
                    score = hit["_normalized_score"] * self.semantic_weight

                    if doc_id in doc_scores:
                        # 이미 키워드 검색에서 나온 경우, 점수 합산
                        doc_scores[doc_id]["score"] += score
                    else:
                        # 시맨틱 검색에서만 나온 경우
                        doc_scores[doc_id] = {"score": score, "hit": hit}
                        initial_doc_ids.add(doc_id)

                # 4. 점수 기준으로 정렬하고 상위 k_per_type개 선택
                sorted_docs = sorted(
                    doc_scores.items(), key=lambda x: x[1]["score"], reverse=True
                )[: self.k_per_type]

                # 5. Document 객체로 변환
                for doc_id, doc_data in sorted_docs:
                    hit = doc_data["hit"]
                    source = hit["_source"]
                    doc = Document(
                        page_content=source.get("page_content", ""),
                        metadata={
                            **source.get("metadata", {}),
                            "hybrid_score": doc_data["score"],
                            "doc_id": doc_id,
                        },
                    )
                    all_documents.append(doc)

            # OpenSearch k-NN을 통한 문서 확장
            if self.use_expansion and initial_doc_ids:
                expanded_docs = self._expand_documents_via_knn(
                    client, list(initial_doc_ids)
                )

                # 중복 제거 (이미 검색된 문서는 제외)
                existing_doc_ids = {doc.metadata.get("doc_id") for doc in all_documents}
                for doc in expanded_docs:
                    doc_id = doc.metadata.get("doc_id")
                    if doc_id and doc_id not in existing_doc_ids:
                        all_documents.append(doc)
                        existing_doc_ids.add(doc_id)

            return all_documents

        except Exception as e:
            print(f"OpenSearch 하이브리드 검색 오류: {e}")
            import traceback

            traceback.print_exc()
            return []
        finally:
            client.close()

    def create_chain(self):
        """체인 생성"""
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
