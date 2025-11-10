"""
OpenSearch에 적재된 임베딩값과 쿼리의 임베딩값을 1:1로 비교하는 테스트 스크립트
"""

import os
import numpy as np
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Tuple

load_dotenv(override=True)

# OpenSearch 설정
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "admin")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "document_embeddings")
OPENSEARCH_EMBEDDING_FIELD = os.getenv("OPENSEARCH_EMBEDDING_FIELD", "embedding")

# 임베딩 모델 설정
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """유클리드 거리 계산"""
    return np.linalg.norm(vec1 - vec2)


def get_opensearch_client() -> OpenSearch:
    """OpenSearch 클라이언트 생성"""
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true",
        verify_certs=False,
        ssl_show_warn=False,
    )


def get_embedding_model() -> OpenAIEmbeddings:
    """임베딩 모델 생성"""
    return OpenAIEmbeddings(
        model=EMBEDDINGS_MODEL_NAME,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
    )


def get_document_embedding(client: OpenSearch, doc_id: str) -> np.ndarray:
    """OpenSearch에서 특정 문서의 임베딩 가져오기"""
    try:
        doc = client.get(
            index=OPENSEARCH_INDEX,
            id=doc_id,
            _source=[OPENSEARCH_EMBEDDING_FIELD, "page_content", "metadata"],
        )
        embedding = doc["_source"].get(OPENSEARCH_EMBEDDING_FIELD)
        if embedding:
            return np.array(embedding, dtype=np.float32)
        else:
            raise ValueError(f"문서 {doc_id}에 임베딩이 없습니다.")
    except Exception as e:
        raise Exception(f"문서 {doc_id} 임베딩 가져오기 실패: {e}")


def get_random_documents(client: OpenSearch, num_docs: int = 5) -> List[Dict]:
    """OpenSearch에서 랜덤 문서 가져오기"""
    try:
        search_body = {
            "size": num_docs,
            "query": {"match_all": {}},
            "_source": ["page_content", "metadata", OPENSEARCH_EMBEDDING_FIELD],
        }
        response = client.search(index=OPENSEARCH_INDEX, body=search_body)
        return [
            {
                "doc_id": hit["_id"],
                "content": hit["_source"].get("page_content", "")[:200],  # 처음 200자만
                "metadata": hit["_source"].get("metadata", {}),
                "embedding": (
                    np.array(
                        hit["_source"].get(OPENSEARCH_EMBEDDING_FIELD), dtype=np.float32
                    )
                    if hit["_source"].get(OPENSEARCH_EMBEDDING_FIELD)
                    else None
                ),
            }
            for hit in response["hits"]["hits"]
        ]
    except Exception as e:
        raise Exception(f"랜덤 문서 가져오기 실패: {e}")


def compare_embeddings(
    query_embedding: np.ndarray,
    doc_embedding: np.ndarray,
    query_text: str = "",
    doc_id: str = "",
) -> Dict:
    """쿼리 임베딩과 문서 임베딩 비교"""
    cosine_sim = cosine_similarity(query_embedding, doc_embedding)
    euclidean_dist = euclidean_distance(query_embedding, doc_embedding)

    return {
        "query_text": query_text,
        "doc_id": doc_id,
        "cosine_similarity": cosine_sim,
        "euclidean_distance": euclidean_dist,
        "query_embedding_shape": query_embedding.shape,
        "doc_embedding_shape": doc_embedding.shape,
        "embeddings_match": np.allclose(query_embedding, doc_embedding, atol=1e-5),
    }


def print_comparison_result(result: Dict, doc_content: str = ""):
    """비교 결과를 보기 좋게 출력"""
    print("=" * 80)
    print(f"📝 쿼리: {result['query_text']}")
    print(f"📄 문서 ID: {result['doc_id']}")
    if doc_content:
        print(f"📄 문서 내용 (처음 200자): {doc_content}")
    print("-" * 80)
    print(f"🔢 코사인 유사도: {result['cosine_similarity']:.6f}")
    print(f"📏 유클리드 거리: {result['euclidean_distance']:.6f}")
    print(f"📐 쿼리 임베딩 차원: {result['query_embedding_shape']}")
    print(f"📐 문서 임베딩 차원: {result['doc_embedding_shape']}")
    print(f"✅ 임베딩 완전 일치: {result['embeddings_match']}")
    print("=" * 80)
    print()


def test_single_comparison(query_text: str, doc_id: str):
    """단일 문서와 쿼리 비교 테스트"""
    print("\n" + "🔍 단일 문서 비교 테스트".center(80, "="))
    print()

    client = get_opensearch_client()
    embedder = get_embedding_model()

    try:
        # 쿼리 임베딩 생성
        print(f"🔄 쿼리 임베딩 생성 중: '{query_text}'")
        query_embedding = np.array(embedder.embed_query(query_text), dtype=np.float32)
        print(f"✅ 쿼리 임베딩 생성 완료 (차원: {query_embedding.shape})")

        # 문서 임베딩 가져오기
        print(f"🔄 문서 임베딩 가져오는 중: {doc_id}")
        doc_data = get_document_embedding(client, doc_id)
        doc_info = client.get(
            index=OPENSEARCH_INDEX,
            id=doc_id,
            _source=["page_content", "metadata"],
        )
        doc_content = doc_info["_source"].get("page_content", "")[:200]
        print(f"✅ 문서 임베딩 가져오기 완료 (차원: {doc_data.shape})")

        # 비교
        result = compare_embeddings(query_embedding, doc_data, query_text, doc_id)
        print_comparison_result(result, doc_content)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
    finally:
        client.close()


def test_multiple_comparisons(query_text: str, num_docs: int = 5):
    """여러 문서와 쿼리 비교 테스트"""
    print("\n" + "🔍 여러 문서 비교 테스트".center(80, "="))
    print()

    client = get_opensearch_client()
    embedder = get_embedding_model()

    try:
        # 쿼리 임베딩 생성
        print(f"🔄 쿼리 임베딩 생성 중: '{query_text}'")
        query_embedding = np.array(embedder.embed_query(query_text), dtype=np.float32)
        print(f"✅ 쿼리 임베딩 생성 완료 (차원: {query_embedding.shape})\n")

        # 랜덤 문서 가져오기
        print(f"🔄 랜덤 문서 {num_docs}개 가져오는 중...")
        documents = get_random_documents(client, num_docs)
        print(f"✅ {len(documents)}개 문서 가져오기 완료\n")

        # 각 문서와 비교
        results = []
        for doc in documents:
            if doc["embedding"] is not None:
                result = compare_embeddings(
                    query_embedding, doc["embedding"], query_text, doc["doc_id"]
                )
                results.append((result, doc["content"]))

        # 유사도 순으로 정렬
        results.sort(key=lambda x: x[0]["cosine_similarity"], reverse=True)

        # 결과 출력
        print(f"\n📊 비교 결과 (코사인 유사도 순):")
        print()
        for i, (result, content) in enumerate(results, 1):
            print(f"\n[{i}] 문서 ID: {result['doc_id']}")
            print(f"    코사인 유사도: {result['cosine_similarity']:.6f}")
            print(f"    유클리드 거리: {result['euclidean_distance']:.6f}")
            print(f"    문서 내용: {content}...")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
    finally:
        client.close()


def test_knn_vs_direct_comparison(query_text: str, top_k: int = 5):
    """OpenSearch k-NN 검색 결과와 직접 비교 결과 비교"""
    print("\n" + "🔍 k-NN 검색 vs 직접 비교 테스트".center(80, "="))
    print()

    client = get_opensearch_client()
    embedder = get_embedding_model()

    try:
        # 쿼리 임베딩 생성
        print(f"🔄 쿼리 임베딩 생성 중: '{query_text}'")
        query_embedding = embedder.embed_query(query_text)
        query_embedding_np = np.array(query_embedding, dtype=np.float32)
        print(f"✅ 쿼리 임베딩 생성 완료 (차원: {query_embedding_np.shape})\n")

        # OpenSearch k-NN 검색
        print("🔄 OpenSearch k-NN 검색 수행 중...")
        knn_query = {
            OPENSEARCH_EMBEDDING_FIELD: {
                "vector": query_embedding,
                "k": top_k,
            }
        }
        search_body = {
            "size": top_k,
            "query": {"knn": knn_query},
            "_source": ["page_content", "metadata", OPENSEARCH_EMBEDDING_FIELD],
        }
        response = client.search(index=OPENSEARCH_INDEX, body=search_body)
        knn_results = response["hits"]["hits"]
        print(f"✅ k-NN 검색 완료 ({len(knn_results)}개 결과)\n")

        # 직접 비교
        print("🔄 직접 비교 수행 중...")
        direct_comparisons = []
        for hit in knn_results:
            doc_id = hit["_id"]
            doc_embedding = np.array(
                hit["_source"].get(OPENSEARCH_EMBEDDING_FIELD), dtype=np.float32
            )
            cosine_sim = cosine_similarity(query_embedding_np, doc_embedding)
            direct_comparisons.append(
                {
                    "doc_id": doc_id,
                    "knn_score": hit["_score"],
                    "direct_cosine": cosine_sim,
                    "content": hit["_source"].get("page_content", "")[:100],
                }
            )
        print("✅ 직접 비교 완료\n")

        # 결과 출력
        print("📊 비교 결과:")
        print("-" * 80)
        print(f"{'순위':<6} {'문서 ID':<30} {'k-NN 점수':<15} {'직접 코사인':<15}")
        print("-" * 80)
        for i, comp in enumerate(direct_comparisons, 1):
            print(
                f"{i:<6} {comp['doc_id']:<30} {comp['knn_score']:<15.6f} {comp['direct_cosine']:<15.6f}"
            )
        print("-" * 80)

        # 차이 분석
        print("\n📈 분석:")
        score_diffs = [
            abs(comp["knn_score"] - comp["direct_cosine"])
            for comp in direct_comparisons
        ]
        print(f"평균 점수 차이: {np.mean(score_diffs):.6f}")
        print(f"최대 점수 차이: {np.max(score_diffs):.6f}")
        print(f"최소 점수 차이: {np.min(score_diffs):.6f}")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
    finally:
        client.close()


def main():
    """메인 함수"""
    print("\n" + "=" * 80)
    print("OpenSearch 임베딩 비교 테스트".center(80))
    print("=" * 80)

    # 테스트 쿼리
    test_query = "word2vec이 뭐야"

    # 테스트 옵션 선택
    print("\n테스트 옵션:")
    print("1. 단일 문서 비교 (문서 ID 필요)")
    print("2. 여러 문서 비교 (랜덤 문서)")
    print("3. k-NN 검색 vs 직접 비교")
    print("4. 모두 실행")

    choice = input("\n선택하세요 (1-4): ").strip()

    if choice == "1":
        doc_id = input("문서 ID를 입력하세요: ").strip()
        if doc_id:
            test_single_comparison(test_query, doc_id)
        else:
            print("❌ 문서 ID가 필요합니다.")

    elif choice == "2":
        num_docs = input("비교할 문서 수를 입력하세요 (기본값: 5): ").strip()
        num_docs = int(num_docs) if num_docs.isdigit() else 5
        test_multiple_comparisons(test_query, num_docs)

    elif choice == "3":
        top_k = input("k-NN 검색 결과 수를 입력하세요 (기본값: 5): ").strip()
        top_k = int(top_k) if top_k.isdigit() else 5
        test_knn_vs_direct_comparison(test_query, top_k)

    elif choice == "4":
        # 랜덤 문서 하나 선택해서 단일 비교
        client = get_opensearch_client()
        docs = get_random_documents(client, 1)
        client.close()
        if docs:
            test_single_comparison(test_query, docs[0]["doc_id"])

        # 여러 문서 비교
        test_multiple_comparisons(test_query, 5)

        # k-NN vs 직접 비교
        test_knn_vs_direct_comparison(test_query, 5)

    else:
        print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()
