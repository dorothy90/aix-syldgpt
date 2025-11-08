"""
Parquet 파일을 syld_gpt OpenSearch 인덱스에 추가하는 스크립트

Parquet 파일 구조:
- 컬럼 0~4095: 임베딩 값 (4096차원)
- 컬럼 4096: 원래 텍스트 (page_content)

사용법:
    python index_parquet_to_syld_gpt.py <parquet_file_path> [--batch-size BATCH_SIZE]

환경 변수:
    OPENSEARCH_HOST: OpenSearch 호스트 (기본값: localhost)
    OPENSEARCH_PORT: OpenSearch 포트 (기본값: 9200)
    OPENSEARCH_USER: OpenSearch 사용자명 (기본값: admin)
    OPENSEARCH_PASSWORD: OpenSearch 비밀번호 (기본값: admin)
    OPENSEARCH_USE_SSL: SSL 사용 여부 (기본값: false)
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from opensearchpy import OpenSearch
import pandas as pd
import numpy as np
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


def check_index_exists(client, index_name: str):
    """인덱스 존재 확인 및 정보 출력"""
    if not client.indices.exists(index=index_name):
        print(f"❌ 오류: '{index_name}' 인덱스가 존재하지 않습니다!")
        return False

    # 인덱스 매핑 확인
    mapping = client.indices.get_mapping(index=index_name)
    index_mapping = mapping[index_name]["mappings"]["properties"]

    print(f"✓ 인덱스 '{index_name}' 확인 완료")

    # embedding 차원 확인
    if "embedding" in index_mapping:
        embedding_dim = index_mapping["embedding"].get("dimension", "알 수 없음")
        print(f"  - 임베딩 차원: {embedding_dim}")

    # 인덱스 통계
    stats = client.indices.stats(index=index_name)
    total_docs = stats["indices"][index_name]["total"]["docs"]["count"]
    print(f"  - 현재 문서 수: {total_docs:,}\n")

    return True


def load_parquet_file(file_path: str):
    """Parquet 파일 로드"""
    print(f"📖 Parquet 파일 로드 중: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"✓ 총 {len(df):,}개 행 로드 완료")
    return df


def extract_embedding_and_text(row, embedding_cols, text_col_idx):
    """
    행에서 임베딩 벡터와 텍스트 추출

    Args:
        row: pandas Series (행 데이터)
        embedding_cols: 임베딩 컬럼 인덱스 리스트 (0~4095)
        text_col_idx: 텍스트 컬럼 인덱스 (4096)

    Returns:
        tuple: (embedding_vector, page_content)
    """
    # 임베딩 벡터 추출 (컬럼 0~4095)
    embedding_vector = row.iloc[embedding_cols].values.tolist()

    # 텍스트 추출 (컬럼 4096)
    page_content = str(row.iloc[text_col_idx])

    return embedding_vector, page_content


def index_parquet_to_syld_gpt(
    parquet_file_path: str,
    index_name: str = "syld_gpt",
    batch_size: int = 1000,
    doc_type: str = "parquet",
    additional_metadata: dict = None,
):
    """
    Parquet 파일을 syld_gpt OpenSearch 인덱스에 추가

    Args:
        parquet_file_path: Parquet 파일 경로
        index_name: OpenSearch 인덱스 이름 (기본값: syld_gpt)
        batch_size: 배치 크기
        doc_type: 문서 타입 (metadata.doc_type에 저장됨, 기본값: "parquet")
        additional_metadata: 추가할 메타데이터 (각 문서에 공통으로 추가됨)
    """
    # OpenSearch 클라이언트 생성
    client = get_opensearch_client()

    # 인덱스 존재 확인
    if not check_index_exists(client, index_name):
        return

    # Parquet 파일 로드
    df = load_parquet_file(parquet_file_path)

    # 컬럼 확인
    print(f"📊 Parquet 파일 컬럼 정보:")
    print(f"   총 컬럼 수: {len(df.columns)}")

    # 임베딩 차원 확인 (컬럼 0~4095 = 4096개)
    embedding_dim = 4096
    text_col_idx = 4096  # 4097번째 컬럼 (0-based index로는 4096)

    # 컬럼이 충분한지 확인
    if len(df.columns) < text_col_idx + 1:
        raise ValueError(
            f"Parquet 파일에 충분한 컬럼이 없습니다. "
            f"필요: {text_col_idx + 1}개, 실제: {len(df.columns)}개"
        )

    # 임베딩 컬럼 인덱스 (0~4095)
    embedding_cols = list(range(embedding_dim))

    # 배치로 인덱싱
    print(f"📤 OpenSearch 인덱스 '{index_name}'에 인덱싱 시작...")
    print(f"   배치 크기: {batch_size}")
    print(f"   doc_type: {doc_type}\n")

    total_rows = len(df)
    indexed_count = 0
    error_count = 0

    # 기본 메타데이터 설정
    default_metadata = {
        "doc_type": doc_type,  # doc_type으로 구분
        "source": "parquet",
        "source_file": str(Path(parquet_file_path).name),
    }
    if additional_metadata:
        default_metadata.update(additional_metadata)

    # 배치 처리
    for start_idx in tqdm(
        range(0, total_rows, batch_size),
        desc="인덱싱 진행",
        total=(total_rows + batch_size - 1) // batch_size,
    ):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]

        # Bulk API용 body 생성
        bulk_body = []

        for idx, (row_idx, row) in enumerate(batch_df.iterrows()):
            try:
                # 임베딩과 텍스트 추출
                embedding_vector, page_content = extract_embedding_and_text(
                    row, embedding_cols, text_col_idx
                )

                # 임베딩 벡터 검증
                if len(embedding_vector) != embedding_dim:
                    print(
                        f"⚠️  경고: 행 {row_idx}의 임베딩 차원이 올바르지 않습니다. "
                        f"예상: {embedding_dim}, 실제: {len(embedding_vector)}"
                    )
                    error_count += 1
                    continue

                # 텍스트가 비어있는지 확인
                if not page_content or page_content.strip() == "":
                    print(f"⚠️  경고: 행 {row_idx}의 텍스트가 비어있습니다.")
                    error_count += 1
                    continue

                # OpenSearch 문서 구조
                doc = {
                    "page_content": page_content,
                    "embedding": embedding_vector,
                    "metadata": default_metadata.copy(),
                }

                # Bulk API 액션 추가
                bulk_body.append({"index": {"_index": index_name}})
                bulk_body.append(doc)

            except Exception as e:
                print(f"⚠️  행 {row_idx} 처리 중 오류: {e}")
                error_count += 1
                continue

        # Bulk API로 인덱싱
        if bulk_body:
            try:
                response = client.bulk(body=bulk_body, refresh=False)

                # 결과 확인
                for item in response.get("items", []):
                    if "index" in item:
                        if item["index"].get("status") in [200, 201]:
                            indexed_count += 1
                        else:
                            error_count += 1
                            if "error" in item["index"]:
                                error_info = item["index"].get("error", {})
                                print(
                                    f"⚠️  인덱싱 오류: {error_info.get('type', 'unknown')} - {error_info.get('reason', 'unknown')}"
                                )

            except Exception as e:
                print(f"⚠️  배치 인덱싱 중 오류: {e}")
                error_count += len(bulk_body) // 2

    # 인덱스 새로고침
    client.indices.refresh(index=index_name)

    # 결과 출력
    print(f"\n✅ 인덱싱 완료!")
    print(f"   총 행 수: {total_rows:,}")
    print(f"   성공: {indexed_count:,}")
    print(f"   실패: {error_count:,}")

    # 인덱스 통계 확인
    stats = client.indices.stats(index=index_name)
    total_docs = stats["indices"][index_name]["total"]["docs"]["count"]
    print(f"   인덱스 총 문서 수: {total_docs:,}")

    # doc_type별 통계 확인
    try:
        search_body = {
            "size": 0,
            "aggs": {
                "doc_types": {"terms": {"field": "metadata.doc_type", "size": 20}}
            },
        }
        agg_response = client.search(index=index_name, body=search_body)
        doc_type_counts = {
            bucket["key"]: bucket["doc_count"]
            for bucket in agg_response["aggregations"]["doc_types"]["buckets"]
        }

        print(f"\n📊 doc_type별 문서 수:")
        for doc_type_name, count in doc_type_counts.items():
            print(f"   - {doc_type_name}: {count:,}개")
    except Exception as e:
        print(f"⚠️  doc_type 통계 조회 중 오류: {e}")

    client.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Parquet 파일을 syld_gpt OpenSearch 인덱스에 추가"
    )
    parser.add_argument(
        "parquet_file",
        type=str,
        help="인덱싱할 Parquet 파일 경로",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="배치 크기 (기본값: 1000)",
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        default="parquet",
        help="문서 타입 (metadata.doc_type에 저장됨, 기본값: parquet)",
    )

    args = parser.parse_args()

    # 파일 존재 확인
    parquet_path = Path(args.parquet_file)
    if not parquet_path.exists():
        print(f"❌ 오류: 파일을 찾을 수 없습니다: {parquet_path}")
        sys.exit(1)

    print("=" * 60)
    print("Parquet 파일 → syld_gpt 인덱스 추가")
    print("=" * 60)
    print(f"파일: {parquet_path}")
    print(f"인덱스: syld_gpt")
    print(f"doc_type: {args.doc_type}")
    print(f"배치 크기: {args.batch_size}\n")

    # 인덱싱 실행
    try:
        index_parquet_to_syld_gpt(
            parquet_file_path=str(parquet_path),
            index_name="syld_gpt",
            batch_size=args.batch_size,
            doc_type=args.doc_type,
        )
        print("\n✅ 작업 완료!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
