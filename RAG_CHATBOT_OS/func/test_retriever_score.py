"""
Retriever 디버깅 도구
- 특정 doc_id가 왜 검색 결과에 나오지 않는지 진단
- 키워드/시맨틱 각각의 rank + raw score 확인
- 현재 retriever.py의 Weighted RRF 로직을 그대로 활용
"""

from retriever import retriever, opensearch_client, text_embedder, OPENSEARCH_INDEX, OPENSEARCH_EMBEDDING_FIELD
from pathlib import Path


# ========== 유틸 ==========
def _find_rank_and_score(hits: list, doc_id: str) -> tuple[int | None, float | None]:
    """검색 결과에서 특정 doc_id의 (1-based rank, raw_score) 반환. 없으면 (None, None)"""
    for i, hit in enumerate(hits):
        if hit["_id"] == doc_id:
            return i + 1, hit["_score"]
    return None, None


def _rrf_score(rank: int | None, rrf_k: int) -> float:
    """RRF 점수: 1/(k + rank). rank가 None이면 0"""
    if rank is None:
        return 0.0
    return 1.0 / (rrf_k + rank)


# ========== 단일 문서 디버깅 ==========
def debug_doc_score(
    query_text: str,
    doc_id: str,
    doc_type_filter: str = None,
    fetch_size: int = 100,
):
    """
    특정 doc_id가 query에 대해 왜 나오는지/안 나오는지 디버깅

    Args:
        query_text: 검색 쿼리
        doc_id: OpenSearch 문서 ID
        doc_type_filter: 문서 타입 필터 (예: "pptx", "pdf")
        fetch_size: 순위 탐색용 검색 크기 (클수록 정확, 기본 100)

    Returns:
        디버깅 결과 dict
    """
    rrf_k = retriever.rrf_k
    kw_weight = retriever.keyword_weight
    sem_weight = retriever.semantic_weight
    top_k = retriever.top_k
    normalized_query = retriever._normalize_text(query_text)

    # 0. 문서 존재 확인
    try:
        doc = opensearch_client.get(
            index=OPENSEARCH_INDEX,
            id=doc_id,
            _source=["page_content", "metadata"],
        )
        source = doc["_source"]
    except Exception as e:
        print(f"[ERROR] 문서 조회 실패 (doc_id={doc_id}): {e}")
        return None

    print("=" * 70)
    print(f"  쿼리        : {query_text.strip()}")
    print(f"  doc_id      : {doc_id}")
    content = source.get("page_content", "")
    print(f"  내용(200자)  : {content[:200]}")
    meta = source.get("metadata", {})
    print(f"  metadata    : {meta}")
    print(f"  설정         : rrf_k={rrf_k}, kw_weight={kw_weight}, sem_weight={sem_weight}, top_k={top_k}")
    print("=" * 70)

    # 1. 키워드(BM25) 검색 — retriever 내부 메서드 직접 사용
    print("\n[1] BM25 키워드 검색")
    kw_hits = retriever._keyword_search(normalized_query, fetch_size, doc_type_filter)
    kw_rank, kw_raw = _find_rank_and_score(kw_hits, doc_id)

    if kw_rank:
        print(f"  rank: {kw_rank}/{len(kw_hits)}  |  raw BM25 score: {kw_raw:.6f}")
    else:
        print(f"  상위 {fetch_size}개에 없음 (키워드 매칭 안 됨)")

    # 2. 시맨틱(kNN) 검색 — retriever 내부 메서드 직접 사용
    print("\n[2] kNN 시맨틱 검색")
    query_embedding = text_embedder.embed_query(normalized_query)
    sem_hits = retriever._semantic_search(query_embedding, fetch_size, doc_type_filter)
    sem_rank, sem_raw = _find_rank_and_score(sem_hits, doc_id)

    if sem_rank:
        print(f"  rank: {sem_rank}/{len(sem_hits)}  |  raw kNN score: {sem_raw:.6f}")
    else:
        print(f"  상위 {fetch_size}개에 없음 (벡터 유사도 낮음)")

    # 3. Weighted RRF 스코어 계산 (retriever._rrf_fusion과 동일 로직)
    kw_rrf = _rrf_score(kw_rank, rrf_k) * kw_weight
    sem_rrf = _rrf_score(sem_rank, rrf_k) * sem_weight
    total_rrf = kw_rrf + sem_rrf

    print(f"\n[3] Weighted RRF 스코어 (k={rrf_k})")
    if kw_rank:
        print(f"  키워드 : {kw_weight} * 1/({rrf_k}+{kw_rank}) = {kw_rrf:.8f}")
    else:
        print(f"  키워드 : 0  (순위 없음)")
    if sem_rank:
        print(f"  시맨틱 : {sem_weight} * 1/({rrf_k}+{sem_rank}) = {sem_rrf:.8f}")
    else:
        print(f"  시맨틱 : 0  (순위 없음)")
    print(f"  ─────────────────────────────")
    print(f"  합산 RRF : {total_rrf:.8f}")

    # 4. 실제 하이브리드 검색 상위 결과와 비교
    actual_fetch = top_k * 3
    doc_scores = retriever._rrf_fusion(
        retriever._keyword_search(normalized_query, actual_fetch, doc_type_filter),
        retriever._semantic_search(query_embedding, actual_fetch, doc_type_filter),
    )
    sorted_all = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    # 대상 문서의 실제 순위 확인
    actual_rank = None
    for i, (did, _) in enumerate(sorted_all, 1):
        if did == doc_id:
            actual_rank = i
            break

    print(f"\n[4] 실제 하이브리드 결과에서의 위치 (fetch_size={actual_fetch})")
    if actual_rank:
        cutoff = "통과" if actual_rank <= top_k else "컷오프"
        print(f"  하이브리드 rank: {actual_rank}/{len(sorted_all)}  →  top_k={top_k} {cutoff}")
    else:
        print(f"  하이브리드 후보 풀({len(sorted_all)}개)에 없음")

    # 5. 상위 top_k 문서 목록 (비교용)
    print(f"\n[5] 현재 top_{top_k} 결과 (이 문서를 이기고 있는 문서들)")
    print(f"  {'rank':>4} | {'doc_id':<36} | {'RRF score':>10} | 내용 앞 80자")
    print(f"  {'-'*4}-+-{'-'*36}-+-{'-'*10}-+-{'-'*40}")
    for i, (did, data) in enumerate(sorted_all[:top_k], 1):
        marker = " ◀" if did == doc_id else ""
        snippet = data["hit"]["_source"].get("page_content", "")[:80].replace("\n", " ")
        print(f"  {i:>4} | {did:<36} | {data['score']:>10.8f} | {snippet}{marker}")

    # 6. 진단 메시지
    print(f"\n[진단]")
    reasons = []
    if kw_rank is None and sem_rank is None:
        reasons.append("키워드/시맨틱 모두 상위 결과에 없음 → 문서 내용이 쿼리와 관련 없거나, 인덱싱 문제 확인 필요")
    else:
        if kw_rank is None:
            reasons.append("키워드 매칭 안 됨 → 쿼리 단어가 문서 본문에 없거나, analyzer 토큰화 차이")
        elif kw_rank and kw_rank > 30:
            reasons.append(f"키워드 rank가 낮음({kw_rank}위) → 쿼리 단어가 문서에 있지만 빈도/관련도 낮음")

        if sem_rank is None:
            reasons.append("시맨틱 매칭 안 됨 → 임베딩 벡터 유사도가 매우 낮음")
        elif sem_rank and sem_rank > 30:
            reasons.append(f"시맨틱 rank가 낮음({sem_rank}위) → 의미적으로는 약하게 관련")

        if actual_rank and actual_rank > top_k:
            reasons.append(
                f"RRF 합산 후 {actual_rank}위 → top_k={top_k} 컷오프에 걸림. "
                f"top_k를 {actual_rank} 이상으로 올리면 포함됨"
            )

    if not reasons:
        reasons.append("이 문서는 현재 설정으로 정상 반환됩니다.")

    for r in reasons:
        print(f"  • {r}")

    print()
    return {
        "doc_id": doc_id,
        "query": query_text.strip(),
        "keyword_rank": kw_rank,
        "keyword_raw_score": kw_raw,
        "keyword_rrf": kw_rrf,
        "semantic_rank": sem_rank,
        "semantic_raw_score": sem_raw,
        "semantic_rrf": sem_rrf,
        "rrf_total": total_rrf,
        "hybrid_rank": actual_rank,
        "in_top_k": actual_rank is not None and actual_rank <= top_k,
    }


# ========== 여러 문서 비교 ==========
def debug_doc_batch(
    query_text: str,
    doc_ids: list[str],
    doc_type_filter: str = None,
    fetch_size: int = 100,
):
    """
    여러 doc_id의 RRF 스코어를 한눈에 비교

    키워드/시맨틱 검색은 1번만 실행하고 각 doc_id의 rank를 찾음
    """
    rrf_k = retriever.rrf_k
    kw_weight = retriever.keyword_weight
    sem_weight = retriever.semantic_weight
    normalized_query = retriever._normalize_text(query_text)

    print("=" * 100)
    print(f"  쿼리: {query_text.strip()}")
    print(f"  비교 문서 수: {len(doc_ids)}")
    print(f"  설정: rrf_k={rrf_k}, kw_weight={kw_weight}, sem_weight={sem_weight}")
    print("=" * 100)

    # 검색 각 1회만
    kw_hits = retriever._keyword_search(normalized_query, fetch_size, doc_type_filter)
    query_embedding = text_embedder.embed_query(normalized_query)
    sem_hits = retriever._semantic_search(query_embedding, fetch_size, doc_type_filter)

    results = []
    for doc_id in doc_ids:
        kw_rank, kw_raw = _find_rank_and_score(kw_hits, doc_id)
        sem_rank, sem_raw = _find_rank_and_score(sem_hits, doc_id)
        kw_rrf = _rrf_score(kw_rank, rrf_k) * kw_weight
        sem_rrf = _rrf_score(sem_rank, rrf_k) * sem_weight
        total = kw_rrf + sem_rrf
        results.append({
            "doc_id": doc_id,
            "kw_rank": kw_rank,
            "kw_raw": kw_raw,
            "sem_rank": sem_rank,
            "sem_raw": sem_raw,
            "kw_rrf": kw_rrf,
            "sem_rrf": sem_rrf,
            "rrf_total": total,
        })

    # RRF 합산 내림차순 정렬
    results.sort(key=lambda x: x["rrf_total"], reverse=True)

    header = f"  {'#':>3} | {'doc_id':<36} | {'kw_rank':>7} | {'kw_raw':>10} | {'sem_rank':>8} | {'sem_raw':>10} | {'kw_rrf':>10} | {'sem_rrf':>10} | {'RRF합산':>10}"
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")
    for i, r in enumerate(results, 1):
        kw_r = str(r["kw_rank"]) if r["kw_rank"] else "-"
        sem_r = str(r["sem_rank"]) if r["sem_rank"] else "-"
        kw_raw_s = f"{r['kw_raw']:.4f}" if r["kw_raw"] is not None else "-"
        sem_raw_s = f"{r['sem_raw']:.4f}" if r["sem_raw"] is not None else "-"
        print(
            f"  {i:>3} | {r['doc_id']:<36} | {kw_r:>7} | {kw_raw_s:>10} | "
            f"{sem_r:>8} | {sem_raw_s:>10} | {r['kw_rrf']:>10.8f} | "
            f"{r['sem_rrf']:>10.8f} | {r['rrf_total']:>10.8f}"
        )

    print()
    return results


# ========== 상위 결과 전체 보기 ==========
def debug_top_results(
    query_text: str,
    top_n: int = 20,
    doc_type_filter: str = None,
):
    """
    쿼리의 상위 top_n개 결과를 키워드/시맨틱 rank와 함께 출력

    내가 기대하는 문서가 왜 안 나오는지, 어떤 문서가 위에 있는지 파악용
    """
    rrf_k = retriever.rrf_k
    kw_weight = retriever.keyword_weight
    sem_weight = retriever.semantic_weight
    normalized_query = retriever._normalize_text(query_text)

    fetch_size = max(top_n * 3, 60)

    kw_hits = retriever._keyword_search(normalized_query, fetch_size, doc_type_filter)
    query_embedding = text_embedder.embed_query(normalized_query)
    sem_hits = retriever._semantic_search(query_embedding, fetch_size, doc_type_filter)

    # RRF 퓨전 (retriever 로직 그대로)
    doc_scores = retriever._rrf_fusion(kw_hits, sem_hits)
    sorted_all = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    # 각 문서의 개별 rank도 역산
    kw_rank_map = {hit["_id"]: (i + 1, hit["_score"]) for i, hit in enumerate(kw_hits)}
    sem_rank_map = {hit["_id"]: (i + 1, hit["_score"]) for i, hit in enumerate(sem_hits)}

    print("=" * 120)
    print(f"  쿼리: {query_text.strip()}")
    print(f"  상위 {top_n}개 | rrf_k={rrf_k}, kw_weight={kw_weight}, sem_weight={sem_weight}")
    print("=" * 120)

    header = f"  {'#':>3} | {'doc_id':<36} | {'kw_rank':>7} | {'sem_rank':>8} | {'RRF':>10} | 내용 앞 100자"
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")

    for i, (doc_id, data) in enumerate(sorted_all[:top_n], 1):
        kw_info = kw_rank_map.get(doc_id)
        sem_info = sem_rank_map.get(doc_id)
        kw_r = str(kw_info[0]) if kw_info else "-"
        sem_r = str(sem_info[0]) if sem_info else "-"
        snippet = data["hit"]["_source"].get("page_content", "")[:100].replace("\n", " ")
        print(
            f"  {i:>3} | {doc_id:<36} | {kw_r:>7} | {sem_r:>8} | "
            f"{data['score']:>10.8f} | {snippet}"
        )

    print()


# ========== 실행 예시 ==========
if __name__ == "__main__":
    # 1) 단일 문서 디버깅
    debug_doc_score(
        query_text="word2vec이 뭐야",
        doc_id="EQUIP팀_2025-50_mail_002_part_1",
    )

    # 2) 여러 문서 비교
    debug_doc_batch(
        query_text="word2vec이 뭐야",
        doc_ids=["doc_id_1", "doc_id_2", "doc_id_3"],
    )

    # 3) 상위 결과 전체 보기
    debug_top_results(
        query_text="word2vec이 뭐야",
        top_n=20,
    )

    pass
