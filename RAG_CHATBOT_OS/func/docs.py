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

    def _keyword_search(self, client, query: str, size: int):
        """BM25 키워드 검색"""
        search_body = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["page_content"],
                    "type": "best_fields",
                }
            },
            "_source": ["page_content", "metadata"],
        }

        response = client.search(index=self.opensearch_index, body=search_body)
        return response["hits"]["hits"]

    def _semantic_search(self, client, query_embedding: list, size: int):
        """kNN 벡터 검색"""
        search_body = {
            "size": size,
            "query": {
                "knn": {
                    self.opensearch_embedding_field: {
                        "vector": query_embedding,
                        "k": size,
                    }
                }
            },
            "_source": ["page_content", "metadata"],
        }

        response = client.search(index=self.opensearch_index, body=search_body)
        return response["hits"]["hits"]

    def search_similar_documents(self, query: str) -> list[Document]:
        """하이브리드 검색 수행 (키워드 30% + 시맨틱 70%)"""
        # 쿼리를 임베딩으로 변환
        embedding_model = self.create_embedding()
        query_embedding = embedding_model.embed_query(query)

        client = self._get_opensearch_client()

        try:
            # 더 많은 결과를 가져와서 재순위화
            fetch_size = self.k * 2

            # 1. 키워드 검색 (BM25)
            keyword_hits = self._keyword_search(client, query, fetch_size)
            keyword_hits = self._normalize_scores(keyword_hits)

            # 2. 시맨틱 검색 (kNN)
            semantic_hits = self._semantic_search(client, query_embedding, fetch_size)
            semantic_hits = self._normalize_scores(semantic_hits)

            # 3. 결과 병합 및 점수 계산
            doc_scores = {}  # {doc_id: {"score": float, "hit": dict}}

            # 키워드 검색 결과 추가 (30% 가중치)
            for hit in keyword_hits:
                doc_id = hit["_id"]
                score = hit["_normalized_score"] * self.keyword_weight
                doc_scores[doc_id] = {"score": score, "hit": hit}

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

            # 4. 점수 기준으로 정렬하고 상위 k개 선택
            sorted_docs = sorted(
                doc_scores.items(), key=lambda x: x[1]["score"], reverse=True
            )[: self.k]

            # 5. Document 객체로 변환
            documents = []
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
                documents.append(doc)

            return documents

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
