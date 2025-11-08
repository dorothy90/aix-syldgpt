# 의존: langchain_community, langchain, opensearch-py, networkx, numpy

# %%
import os
import json
import pickle
from operator import itemgetter
from typing import Optional

from dotenv import load_dotenv
from opensearchpy import OpenSearch
import networkx as nx
import numpy as np

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

        # NetworkX 그래프 설정
        self.use_graphdb = os.getenv("USE_GRAPHDB", "false").lower() == "true"
        self.graphdb_file = os.getenv(
            "GRAPHDB_FILE", "document_graph.pkl"
        )  # 그래프 파일 경로
        self.graphdb_expansion_limit = int(os.getenv("GRAPHDB_EXPANSION_LIMIT", "10"))
        self.graphdb_max_depth = int(os.getenv("GRAPHDB_MAX_DEPTH", "1"))

        # NetworkX 그래프 초기화
        self.doc_graph: Optional[nx.Graph] = None
        if self.use_graphdb:
            self._load_document_graph()

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
        """BM25 키워드 검색"""
        query_clause = {
            "multi_match": {
                "query": query,
                "fields": ["page_content"],
                "type": "best_fields",
            }
        }

        # doc_type 필터 추가
        if doc_type_filter is not None:
            if doc_type_filter == "parquet":
                # parquet인 경우
                query_clause = {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["page_content"],
                                    "type": "best_fields",
                                }
                            },
                            {"term": {"metadata.doc_type": "parquet"}},
                        ]
                    }
                }
            else:
                # parquet이 아닌 경우
                query_clause = {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["page_content"],
                                    "type": "best_fields",
                                }
                            },
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

    def _load_document_graph(self):
        """문서 관계 그래프 로드"""
        try:
            if self.graphdb_file and os.path.exists(self.graphdb_file):
                if self.graphdb_file.endswith(".pkl") or self.graphdb_file.endswith(
                    ".pickle"
                ):
                    with open(self.graphdb_file, "rb") as f:
                        self.doc_graph = pickle.load(f)
                elif self.graphdb_file.endswith(".json"):
                    with open(self.graphdb_file, "r", encoding="utf-8") as f:
                        graph_data = json.load(f)
                        self.doc_graph = nx.node_link_graph(graph_data)
                elif self.graphdb_file.endswith(".graphml"):
                    self.doc_graph = nx.read_graphml(self.graphdb_file)
                else:
                    print(f"지원하지 않는 그래프 파일 형식: {self.graphdb_file}")
                    self.doc_graph = nx.Graph()
            else:
                print(f"그래프 파일을 찾을 수 없습니다: {self.graphdb_file}")
                self.doc_graph = nx.Graph()

            if self.doc_graph is None:
                self.doc_graph = nx.Graph()

            print(
                f"문서 그래프 로드 완료: {self.doc_graph.number_of_nodes()}개 노드, {self.doc_graph.number_of_edges()}개 엣지"
            )
        except Exception as e:
            print(f"그래프 로드 오류: {e}")
            import traceback

            traceback.print_exc()
            self.doc_graph = nx.Graph()

    def _expand_documents_via_graphdb(self, doc_ids: list[str]) -> list[str]:
        """NetworkX 그래프를 통해 연결된 문서 ID들을 찾기"""
        if not self.use_graphdb or not self.doc_graph or not doc_ids:
            return []

        expanded_doc_ids = set()

        try:
            for doc_id in doc_ids:
                if doc_id in self.doc_graph:
                    if self.graphdb_max_depth > 1:
                        # BFS로 다단계 탐색
                        visited = {doc_id}
                        queue = [(doc_id, 0)]

                        while queue:
                            current_node, depth = queue.pop(0)

                            if depth >= self.graphdb_max_depth:
                                continue

                            for neighbor in self.doc_graph.neighbors(current_node):
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    if neighbor not in doc_ids:
                                        expanded_doc_ids.add(neighbor)
                                    queue.append((neighbor, depth + 1))
                    else:
                        # 직접 이웃만
                        for neighbor in self.doc_graph.neighbors(doc_id):
                            if neighbor not in doc_ids:
                                expanded_doc_ids.add(neighbor)

            # 확장 제한 적용
            expanded_list = list(expanded_doc_ids)
            if len(expanded_list) > self.graphdb_expansion_limit:
                # 연결 수가 많은 노드 우선
                expanded_list = sorted(
                    expanded_list,
                    key=lambda x: (
                        self.doc_graph.degree(x) if x in self.doc_graph else 0
                    ),
                    reverse=True,
                )[: self.graphdb_expansion_limit]

            return expanded_list

        except Exception as e:
            print(f"GraphDB 확장 검색 오류: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _get_documents_by_ids(self, client, doc_ids: list[str]) -> list[Document]:
        """OpenSearch에서 문서 ID로 문서 가져오기"""
        if not doc_ids:
            return []

        try:
            search_body = {
                "query": {"ids": {"values": doc_ids}},
                "size": len(doc_ids),
                "_source": ["page_content", "metadata"],
            }

            response = client.search(index=self.opensearch_index, body=search_body)
            documents = []

            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                doc = Document(
                    page_content=source.get("page_content", ""),
                    metadata={
                        **source.get("metadata", {}),
                        "doc_id": hit["_id"],
                        "graphdb_expanded": True,
                    },
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"문서 ID로 검색 오류: {e}")
            return []

    def search_similar_documents(self, query: str) -> list[Document]:
        """하이브리드 검색 수행 (키워드 30% + 시맨틱 70%)
        doc_type이 parquet인 것과 아닌 것을 각각 k_per_type개씩 검색
        NetworkX 그래프를 통해 추가 문서 확장"""
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

            # NetworkX 그래프를 통한 문서 확장
            if self.use_graphdb and initial_doc_ids:
                expanded_doc_ids = self._expand_documents_via_graphdb(
                    list(initial_doc_ids)
                )

                if expanded_doc_ids:
                    # 확장된 문서들을 OpenSearch에서 가져오기
                    expanded_docs = self._get_documents_by_ids(client, expanded_doc_ids)

                    # 중복 제거 (이미 검색된 문서는 제외)
                    existing_doc_ids = {
                        doc.metadata.get("doc_id") for doc in all_documents
                    }
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
