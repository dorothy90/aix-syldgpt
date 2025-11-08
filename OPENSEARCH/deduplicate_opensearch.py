"""
OpenSearch 인덱스에서 page_content 기준 중복 제거 스크립트

사용법:
    python deduplicate_opensearch.py

환경 변수:
    OPENSEARCH_HOST: OpenSearch 호스트 (기본값: localhost)
    OPENSEARCH_PORT: OpenSearch 포트 (기본값: 9200)
    OPENSEARCH_USER: OpenSearch 사용자명 (기본값: admin)
    OPENSEARCH_PASSWORD: OpenSearch 비밀번호 (기본값: admin)
    OPENSEARCH_USE_SSL: SSL 사용 여부 (기본값: false)
"""

import os
import hashlib
from collections import defaultdict
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from tqdm import tqdm

load_dotenv(override=True)


def get_opensearch_client():
    """OpenSearch 클라이언트 생성"""
    return OpenSearch(
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


def scan_all_documents(client, index_name: str, batch_size: int = 1000):
    """
    OpenSearch 인덱스의 모든 문서를 스캔하여 반환

    Args:
        client: OpenSearch 클라이언트
        index_name: 인덱스 이름
        batch_size: 한 번에 가져올 문서 수

    Yields:
        문서 딕셔너리 (doc_id, page_content 포함)
    """
    # Scroll API를 사용하여 모든 문서 가져오기
    search_body = {
        "size": batch_size,
        "_source": ["page_content", "embedding"],
    }

    response = client.search(
        index=index_name, body=search_body, scroll="5m"  # 스크롤 컨텍스트 유지 시간
    )

    scroll_id = response.get("_scroll_id")
    hits = response["hits"]["hits"]

    # 첫 번째 배치 처리
    for hit in hits:
        yield {
            "doc_id": hit["_id"],
            "page_content": hit["_source"].get("page_content", ""),
            "embedding": hit["_source"].get("embedding"),
        }

    # 나머지 문서들 스크롤하여 가져오기
    while len(hits) > 0:
        response = client.scroll(scroll_id=scroll_id, scroll="5m")
        scroll_id = response.get("_scroll_id")
        hits = response["hits"]["hits"]

        for hit in hits:
            yield {
                "doc_id": hit["_id"],
                "page_content": hit["_source"].get("page_content", ""),
                "embedding": hit["_source"].get("embedding"),
            }

    # 스크롤 컨텍스트 정리
    if scroll_id:
        client.clear_scroll(scroll_id=scroll_id)


def get_content_hash(page_content: str) -> str:
    """
    page_content의 해시값 계산 (메모리 효율적 중복 검사용)

    SHA-256 해시를 사용하여 충돌 가능성을 거의 없앰
    """
    normalized_content = page_content.strip()
    return hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()


def find_duplicates(client, index_name: str):
    """
    page_content 기준으로 중복 문서 찾기 (메모리 효율적 버전)

    중요:
    - page_content의 해시값만 메모리에 저장하여 메모리 사용량을 최소화합니다
    - 모든 문서를 한 번에 스캔하여 맨 앞과 맨 뒤 문서의 중복도 정확히 찾을 수 있습니다
    - 해시 충돌 가능성은 거의 없지만, 필요시 실제 내용 비교로 검증 가능

    Returns:
        dict: {page_content_hash: [doc_ids]} 형태의 딕셔너리
    """
    print(f"📖 '{index_name}' 인덱스의 모든 문서를 스캔하는 중...")
    print("   💡 메모리 효율적 방식: page_content의 해시값만 저장합니다")
    print("   → 맨 앞과 맨 뒤 문서의 중복도 정확히 찾을 수 있습니다.\n")

    # 해시값을 키로 하는 딕셔너리 (메모리 효율적)
    # 실제 page_content 대신 해시값만 저장하여 메모리 사용량을 크게 줄임
    hash_to_docs = defaultdict(list)

    # 모든 문서를 순차적으로 스캔
    doc_count = 0
    for doc in scan_all_documents(client, index_name):
        doc_count += 1
        page_content = doc["page_content"]

        # 해시값 계산 (메모리 효율적)
        content_hash = get_content_hash(page_content)
        hash_to_docs[content_hash].append(doc["doc_id"])

        # 진행 상황 표시 (10000개마다)
        if doc_count % 10000 == 0:
            print(
                f"   진행 중... {doc_count:,}개 문서 스캔 완료 (메모리 사용량 최적화)"
            )

    print(f"   ✅ 총 {doc_count:,}개 문서 스캔 완료\n")

    # 중복이 있는 항목만 필터링
    duplicates = {
        content_hash: doc_ids
        for content_hash, doc_ids in hash_to_docs.items()
        if len(doc_ids) > 1
    }

    print(f"   📊 고유한 page_content 해시 개수: {len(hash_to_docs):,}")
    print(f"   📊 중복된 page_content 해시 개수: {len(duplicates):,}\n")

    return duplicates


def remove_duplicates(
    client,
    index_name: str,
    keep_strategy: str = "first",
    dry_run: bool = True,
):
    """
    중복 문서 제거

    Args:
        client: OpenSearch 클라이언트
        index_name: 인덱스 이름
        keep_strategy: 유지할 문서 선택 전략
            - "first": 각 그룹의 첫 번째 문서 유지 (기본값)
            - "last": 각 그룹의 마지막 문서 유지
        dry_run: True이면 실제 삭제하지 않고 시뮬레이션만 수행
    """
    # 중복 찾기
    duplicates = find_duplicates(client, index_name)

    if not duplicates:
        print("✅ 중복 문서가 없습니다!")
        return

    total_duplicates = sum(len(doc_ids) - 1 for doc_ids in duplicates.values())
    total_unique_contents = len(duplicates)

    print(f"\n📊 중복 분석 결과:")
    print(f"   - 중복된 page_content 개수: {total_unique_contents:,}")
    print(f"   - 삭제될 문서 개수: {total_duplicates:,}")

    if dry_run:
        print("\n⚠️  DRY RUN 모드: 실제로 삭제하지 않습니다.")
        print("   실제 삭제를 원하면 dry_run=False로 설정하세요.\n")
    else:
        print("\n🗑️  실제 삭제 모드를 시작합니다...\n")

    # 삭제할 문서 ID 수집
    docs_to_delete = []

    for content, doc_ids in tqdm(
        duplicates.items(), desc="중복 문서 처리", total=len(duplicates)
    ):
        if keep_strategy == "first":
            # 첫 번째 문서 유지, 나머지 삭제
            keep_id = doc_ids[0]
            delete_ids = doc_ids[1:]
        elif keep_strategy == "last":
            # 마지막 문서 유지, 나머지 삭제
            keep_id = doc_ids[-1]
            delete_ids = doc_ids[:-1]
        else:
            raise ValueError(f"알 수 없는 전략: {keep_strategy}")

        docs_to_delete.extend(delete_ids)

    if not docs_to_delete:
        print("✅ 삭제할 문서가 없습니다!")
        return

    # 실제 삭제 수행
    if not dry_run:
        print(f"\n🗑️  {len(docs_to_delete):,}개의 중복 문서를 삭제하는 중...")

        # 배치로 삭제 (성능 향상)
        batch_size = 1000
        deleted_count = 0

        for i in tqdm(
            range(0, len(docs_to_delete), batch_size),
            desc="문서 삭제",
            total=(len(docs_to_delete) + batch_size - 1) // batch_size,
        ):
            batch = docs_to_delete[i : i + batch_size]

            # Bulk API를 사용하여 배치 삭제
            body = []
            for doc_id in batch:
                body.append({"delete": {"_index": index_name, "_id": doc_id}})

            response = client.bulk(body=body)

            # 삭제된 문서 수 카운트
            for item in response.get("items", []):
                if "delete" in item and item["delete"].get("status") in [200, 404]:
                    deleted_count += 1

        print(f"\n✅ 완료! {deleted_count:,}개의 중복 문서가 삭제되었습니다.")
    else:
        print(f"\n📋 시뮬레이션 결과:")
        print(f"   - 삭제될 문서 ID 개수: {len(docs_to_delete):,}")
        print(f"   - 유지될 문서 ID 개수: {total_unique_contents:,}")

        # 샘플 출력 (처음 5개)
        print("\n📝 삭제될 문서 샘플 (처음 5개):")
        for i, doc_id in enumerate(docs_to_delete[:5]):
            print(f"   {i+1}. {doc_id}")
        if len(docs_to_delete) > 5:
            print(f"   ... 외 {len(docs_to_delete) - 5}개")


def main():
    """메인 함수"""
    index_name = "syld_gpt"

    print("=" * 60)
    print("OpenSearch 중복 제거 스크립트")
    print("=" * 60)
    print(f"인덱스: {index_name}\n")

    # OpenSearch 클라이언트 생성
    client = get_opensearch_client()

    # 인덱스 존재 확인
    if not client.indices.exists(index=index_name):
        print(f"❌ 오류: '{index_name}' 인덱스가 존재하지 않습니다!")
        return

    # 인덱스 통계 확인
    stats = client.indices.stats(index=index_name)
    total_docs = stats["indices"][index_name]["total"]["docs"]["count"]
    print(f"📊 현재 인덱스 문서 수: {total_docs:,}\n")

    # 먼저 dry_run으로 확인
    print("=" * 60)
    print("1단계: 중복 분석 (DRY RUN)")
    print("=" * 60)
    remove_duplicates(
        client=client,
        index_name=index_name,
        keep_strategy="first",
        dry_run=True,
    )

    # 사용자 확인
    print("\n" + "=" * 60)
    response = input("실제로 중복 문서를 삭제하시겠습니까? (yes/no): ").strip().lower()

    if response == "yes":
        print("\n" + "=" * 60)
        print("2단계: 실제 삭제 수행")
        print("=" * 60)
        remove_duplicates(
            client=client,
            index_name=index_name,
            keep_strategy="first",
            dry_run=False,
        )

        # 최종 통계
        stats_after = client.indices.stats(index=index_name)
        total_docs_after = stats_after["indices"][index_name]["total"]["docs"]["count"]
        print(f"\n📊 삭제 후 인덱스 문서 수: {total_docs_after:,}")
        print(f"📉 삭제된 문서 수: {total_docs - total_docs_after:,}")
    else:
        print("\n❌ 삭제 작업이 취소되었습니다.")

    client.close()
    print("\n✅ 작업 완료!")


if __name__ == "__main__":
    main()
