"""
OpenSearch의 임베딩을 기반으로 문서 관계 그래프 생성
"""

import os
import json
import pickle
import numpy as np
from dotenv import load_dotenv
from opensearchpy import OpenSearch
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

load_dotenv(override=True)

# OpenSearch 설정
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "admin")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "document_embeddings")
OPENSEARCH_EMBEDDING_FIELD = os.getenv("OPENSEARCH_EMBEDDING_FIELD", "embedding")

# 그래프 생성 설정
SIMILARITY_THRESHOLD = float(
    os.getenv("GRAPH_SIMILARITY_THRESHOLD", "0.7")
)  # 유사도 임계값
TOP_K_NEIGHBORS = int(
    os.getenv("GRAPH_TOP_K_NEIGHBORS", "10")
)  # 각 문서당 최대 연결 수
GRAPH_OUTPUT_FILE = os.getenv(
    "GRAPH_OUTPUT_FILE", "document_graph.pkl"
)  # 출력 파일 경로
# 병렬 처리 설정
MAX_WORKERS = int(os.getenv("GRAPH_MAX_WORKERS", "10"))  # 병렬 처리 스레드 수


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def get_opensearch_client():
    """OpenSearch 클라이언트 생성"""
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true",
        verify_certs=False,
        ssl_show_warn=False,
    )


def fetch_all_document_ids(client: OpenSearch) -> list[str]:
    """OpenSearch에서 모든 문서의 ID만 가져오기 (메모리 효율적)"""
    doc_ids = []

    try:
        # 스크롤을 사용하여 모든 문서 ID 가져오기
        search_body = {
            "size": 1000,  # 한 번에 가져올 문서 수
            "query": {"match_all": {}},
            "_source": False,  # ID만 가져오기
        }

        response = client.search(index=OPENSEARCH_INDEX, body=search_body, scroll="5m")

        scroll_id = response.get("_scroll_id")
        hits = response["hits"]["hits"]

        # 첫 번째 배치 처리
        for hit in hits:
            doc_ids.append(hit["_id"])

        # 스크롤로 나머지 문서 가져오기
        while scroll_id and hits:
            response = client.scroll(scroll_id=scroll_id, scroll="5m")
            hits = response["hits"]["hits"]
            scroll_id = response.get("_scroll_id")

            for hit in hits:
                doc_ids.append(hit["_id"])

        # 스크롤 정리
        if scroll_id:
            client.clear_scroll(scroll_id=scroll_id)

        print(f"총 {len(doc_ids)}개 문서 ID 로드 완료")
        return doc_ids

    except Exception as e:
        print(f"문서 ID 로드 오류: {e}")
        import traceback

        traceback.print_exc()
        return []


def fetch_all_documents(client: OpenSearch) -> list[dict]:
    """OpenSearch에서 모든 문서의 ID와 임베딩 가져오기 (기존 함수 유지)"""
    documents = []

    try:
        # 스크롤을 사용하여 모든 문서 가져오기
        search_body = {
            "size": 1000,  # 한 번에 가져올 문서 수
            "query": {"match_all": {}},
            "_source": [OPENSEARCH_EMBEDDING_FIELD],  # 임베딩만 가져오기
        }

        response = client.search(
            index=OPENSEARCH_INDEX, body=search_body, scroll="5m"  # 스크롤 유지 시간
        )

        scroll_id = response.get("_scroll_id")
        hits = response["hits"]["hits"]

        # 첫 번째 배치 처리
        for hit in hits:
            doc_id = hit["_id"]
            embedding = hit["_source"].get(OPENSEARCH_EMBEDDING_FIELD)
            if embedding:
                documents.append(
                    {"id": doc_id, "embedding": np.array(embedding, dtype=np.float32)}
                )

        # 스크롤로 나머지 문서 가져오기
        while scroll_id and hits:
            response = client.scroll(scroll_id=scroll_id, scroll="5m")
            hits = response["hits"]["hits"]
            scroll_id = response.get("_scroll_id")

            for hit in hits:
                doc_id = hit["_id"]
                embedding = hit["_source"].get(OPENSEARCH_EMBEDDING_FIELD)
                if embedding:
                    documents.append(
                        {
                            "id": doc_id,
                            "embedding": np.array(embedding, dtype=np.float32),
                        }
                    )

        # 스크롤 정리
        if scroll_id:
            client.clear_scroll(scroll_id=scroll_id)

        print(f"총 {len(documents)}개 문서 로드 완료")
        return documents

    except Exception as e:
        print(f"문서 로드 오류: {e}")
        import traceback

        traceback.print_exc()
        return []


def build_graph_using_opensearch_knn(
    client: OpenSearch,
    doc_ids: list[str],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    top_k: int = TOP_K_NEIGHBORS,
    max_workers: Optional[int] = None,
) -> nx.Graph:
    """
    OpenSearch의 k-NN 검색을 사용하여 그래프 생성 (최적화된 버전)

    Args:
        client: OpenSearch 클라이언트
        doc_ids: 문서 ID 리스트
        similarity_threshold: 엣지를 생성할 최소 유사도 임계값
        top_k: 각 문서당 최대 연결 수
        max_workers: 병렬 처리 스레드 수 (None이면 MAX_WORKERS 사용)

    Returns:
        NetworkX 그래프 객체
    """
    graph = nx.Graph()

    if max_workers is None:
        max_workers = MAX_WORKERS

    print(f"\n그래프 생성 시작 (OpenSearch k-NN 사용)...")
    print(f"  - 문서 수: {len(doc_ids)}")
    print(f"  - 유사도 임계값: {similarity_threshold}")
    print(f"  - 문서당 최대 연결 수: {top_k}")
    print(f"  - 병렬 처리 스레드 수: {max_workers}")

    # 모든 문서를 노드로 추가
    for doc_id in doc_ids:
        graph.add_node(doc_id)

    def process_document(doc_id: str) -> list[tuple[str, float]]:
        """단일 문서에 대한 유사 문서 찾기"""
        try:
            # 해당 문서의 임베딩 가져오기
            doc = client.get(
                index=OPENSEARCH_INDEX,
                id=doc_id,
                _source=[OPENSEARCH_EMBEDDING_FIELD],
            )
            embedding = doc["_source"].get(OPENSEARCH_EMBEDDING_FIELD)

            if not embedding:
                return []

            # k-NN 검색으로 유사한 문서 찾기
            # top_k보다 더 많이 가져와서 임계값 필터링 후 상위 k개 선택
            search_k = max(top_k * 2, 50)  # 최소 50개는 가져오기

            search_body = {
                "size": search_k,
                "query": {
                    "knn": {
                        OPENSEARCH_EMBEDDING_FIELD: {
                            "vector": embedding,
                            "k": search_k,
                        }
                    }
                },
                "_source": False,  # ID만 필요
            }

            response = client.search(index=OPENSEARCH_INDEX, body=search_body)

            # 유사도가 임계값 이상인 문서만 필터링
            edges = []
            for hit in response["hits"]["hits"]:
                neighbor_id = hit["_id"]
                # OpenSearch의 코사인 유사도는 _score로 반환됨
                # cosinesimil 공간 타입 사용 시 이미 정규화된 값
                similarity = hit["_score"]

                if similarity >= similarity_threshold and neighbor_id != doc_id:
                    edges.append((neighbor_id, similarity))

            # 상위 k개만 선택
            edges.sort(key=lambda x: x[1], reverse=True)
            return edges[:top_k]

        except Exception as e:
            print(f"  ⚠️  문서 {doc_id} 처리 오류: {e}")
            return []

    # 병렬 처리로 각 문서에 대한 유사 문서 찾기
    processed = 0
    total = len(doc_ids)

    if max_workers > 1:
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc_id = {
                executor.submit(process_document, doc_id): doc_id for doc_id in doc_ids
            }

            for future in as_completed(future_to_doc_id):
                doc_id = future_to_doc_id[future]
                try:
                    edges = future.result()
                    for neighbor_id, similarity in edges:
                        graph.add_edge(doc_id, neighbor_id, weight=similarity)

                    processed += 1
                    if processed % 100 == 0:
                        print(
                            f"  진행률: {processed}/{total} ({processed*100//total}%)"
                        )
                except Exception as e:
                    print(f"  ⚠️  문서 {doc_id} 처리 중 오류: {e}")
    else:
        # 순차 처리
        for doc_id in doc_ids:
            edges = process_document(doc_id)
            for neighbor_id, similarity in edges:
                graph.add_edge(doc_id, neighbor_id, weight=similarity)

            processed += 1
            if processed % 100 == 0:
                print(f"  진행률: {processed}/{total} ({processed*100//total}%)")

    print(f"\n그래프 생성 완료!")
    print(f"  - 노드 수: {graph.number_of_nodes()}")
    print(f"  - 엣지 수: {graph.number_of_edges()}")
    if graph.number_of_nodes() > 0:
        print(
            f"  - 평균 연결 수: {graph.number_of_edges() * 2 / graph.number_of_nodes():.2f}"
        )

    return graph


def build_graph_from_embeddings(
    documents: list[dict],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    top_k: int = TOP_K_NEIGHBORS,
) -> nx.Graph:
    """
    임베딩 유사도를 기반으로 그래프 생성 (기존 함수 유지 - 호환성)

    Args:
        documents: [{"id": str, "embedding": np.ndarray}, ...] 형태의 문서 리스트
        similarity_threshold: 엣지를 생성할 최소 유사도 임계값
        top_k: 각 문서당 최대 연결 수 (임계값보다 높아도 상위 k개만 선택)

    Returns:
        NetworkX 그래프 객체
    """
    graph = nx.Graph()

    print(f"\n그래프 생성 시작...")
    print(f"  - 문서 수: {len(documents)}")
    print(f"  - 유사도 임계값: {similarity_threshold}")
    print(f"  - 문서당 최대 연결 수: {top_k}")

    # 모든 문서를 노드로 추가
    for doc in documents:
        graph.add_node(doc["id"])

    # 각 문서 쌍에 대해 유사도 계산
    total_pairs = len(documents) * (len(documents) - 1) // 2
    processed = 0

    for i, doc1 in enumerate(documents):
        # 각 문서에 대해 유사도가 높은 문서들을 찾기
        similarities = []

        for j, doc2 in enumerate(documents):
            if i >= j:  # 중복 계산 방지
                continue

            # 코사인 유사도 계산
            similarity = cosine_similarity(doc1["embedding"], doc2["embedding"])

            if similarity >= similarity_threshold:
                similarities.append((doc2["id"], similarity))

            processed += 1
            if processed % 1000 == 0:
                print(
                    f"  진행률: {processed}/{total_pairs} ({processed*100//total_pairs}%)"
                )

        # 상위 k개만 선택하여 엣지 추가
        similarities.sort(key=lambda x: x[1], reverse=True)
        for neighbor_id, similarity in similarities[:top_k]:
            graph.add_edge(doc1["id"], neighbor_id, weight=similarity)

    print(f"\n그래프 생성 완료!")
    print(f"  - 노드 수: {graph.number_of_nodes()}")
    print(f"  - 엣지 수: {graph.number_of_edges()}")
    if graph.number_of_nodes() > 0:
        print(
            f"  - 평균 연결 수: {graph.number_of_edges() * 2 / graph.number_of_nodes():.2f}"
        )

    return graph


def save_graph(graph: nx.Graph, output_file: str):
    """그래프를 파일로 저장"""
    try:
        if output_file.endswith(".pkl") or output_file.endswith(".pickle"):
            # Pickle 형식 (바이너리, 빠름)
            with open(output_file, "wb") as f:
                pickle.dump(graph, f)
            print(f"\n그래프 저장 완료: {output_file} (Pickle 형식)")

        elif output_file.endswith(".json"):
            # JSON 형식 (텍스트, 호환성 좋음)
            graph_data = nx.node_link_data(graph)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            print(f"\n그래프 저장 완료: {output_file} (JSON 형식)")

        elif output_file.endswith(".graphml"):
            # GraphML 형식 (표준 형식)
            nx.write_graphml(graph, output_file)
            print(f"\n그래프 저장 완료: {output_file} (GraphML 형식)")

        else:
            # 기본값: Pickle
            output_file_pkl = output_file + ".pkl"
            with open(output_file_pkl, "wb") as f:
                pickle.dump(graph, f)
            print(f"\n그래프 저장 완료: {output_file_pkl} (Pickle 형식)")

    except Exception as e:
        print(f"그래프 저장 오류: {e}")
        import traceback

        traceback.print_exc()


def main():
    """메인 함수"""
    print("=" * 60)
    print("문서 관계 그래프 생성 시작 (OpenSearch k-NN 최적화 버전)")
    print("=" * 60)

    # OpenSearch 클라이언트 생성
    client = get_opensearch_client()

    try:
        # 모든 문서 ID만 가져오기 (메모리 효율적)
        doc_ids = fetch_all_document_ids(client)

        if not doc_ids:
            print("⚠️  문서가 없습니다.")
            return

        # OpenSearch k-NN 검색을 사용하여 그래프 생성
        graph = build_graph_using_opensearch_knn(
            client,
            doc_ids,
            similarity_threshold=SIMILARITY_THRESHOLD,
            top_k=TOP_K_NEIGHBORS,
        )

        # 그래프 저장
        save_graph(graph, GRAPH_OUTPUT_FILE)

        print("\n" + "=" * 60)
        print("✅ 그래프 생성 완료!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()

    finally:
        client.close()


if __name__ == "__main__":
    main()
