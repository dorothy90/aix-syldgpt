from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """
    CSV를 DictReader로 읽어서 각 row를 dict로 만든 list[dict] 반환
    """
    if not path.exists():
        return [{"error": "file_error", "detail": f"CSV not found: {path}"}]

    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    except Exception as e:
        return [{"error": "file_error", "detail": str(e)}]


def get_lot_status(lot_id: str) -> List[Dict[str, Any]]:
    """
    LOT 현재 상태 조회 (CSV 기반)
    - 성공: 0~1개 row를 담은 list[dict]
    - 실패: 항상 list[dict] 형태로 error 반환
    """
    lot_id = (lot_id or "").strip()
    rows = _read_csv_rows(_DATA_DIR / "lot_status.csv")
    if rows and isinstance(rows[0], dict) and rows[0].get("error"):
        return rows

    match = next(
        (r for r in rows if (str(r.get("lot_id") or "").strip() == lot_id)),
        None,
    )
    if match is None:
        return [{"error": "not_found", "lot_id": lot_id}]
    return [match]


def get_lot_history(
    lot_id: str,
    filters: Optional[List[Dict[str, Any]]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    LOT 이력 조회 (CSV 기반)
    - 성공: 0~N개 이벤트 row를 담은 list[dict]
    - 실패: 항상 list[dict] 형태로 error 반환
    """
    lot_id = (lot_id or "").strip()
    rows = _read_csv_rows(_DATA_DIR / "lot_history.csv")
    if rows and isinstance(rows[0], dict) and rows[0].get("error"):
        return rows

    history = [r for r in rows if (str(r.get("lot_id") or "").strip() == lot_id)]
    if not history:
        return [{"error": "not_found", "lot_id": lot_id}]

    def _sort_key(r: Dict[str, Any]) -> str:
        return str(r.get("event_time") or "").strip()

    # 정렬(시간 오름차순)
    history = sorted(history, key=_sort_key)

    # -------------------- 결정론 필터 엔진 --------------------
    # 가드: 필터 개수/limit 상한
    max_filters = 10
    max_limit = 500
    if filters is None:
        filters = []
    if len(filters) > max_filters:
        return [
            {
                "error": "bad_filter",
                "detail": f"Too many filters (max={max_filters})",
                "lot_id": lot_id,
            }
        ]

    if limit is not None:
        try:
            limit_int = int(limit)
        except Exception:
            return [{"error": "bad_filter", "detail": "limit must be int", "lot_id": lot_id}]
        if limit_int <= 0:
            limit = None
        else:
            limit = min(limit_int, max_limit)

    allowed_fields = set(history[0].keys()) if history else set()
    allowed_ops = {"eq", "contains", "in", "gte", "lte"}

    def _norm_text(v: Any) -> str:
        return str(v or "").strip()

    def _norm_ci(v: Any) -> str:
        return _norm_text(v).upper()

    def _check_one(row: Dict[str, Any], cond: Dict[str, Any]) -> bool:
        field = _norm_text(cond.get("field"))
        op = _norm_text(cond.get("op") or "eq").lower()
        value = cond.get("value")

        if not field or field not in allowed_fields:
            # field가 없거나 허용되지 않으면 false(결과를 비워버리기보다 명확히 에러 처리)
            raise ValueError(f"Unknown field: {field}")
        if op not in allowed_ops:
            raise ValueError(f"Unknown op: {op}")

        cell = row.get(field)

        if op == "eq":
            return _norm_ci(cell) == _norm_ci(value)
        if op == "contains":
            return _norm_ci(value) in _norm_ci(cell)
        if op == "in":
            items = value if isinstance(value, list) else [value]
            items_norm = {_norm_ci(x) for x in items}
            return _norm_ci(cell) in items_norm
        if op == "gte":
            # event_time이 YYYY-MM-DD HH:MM:SS 형태면 문자열 비교로도 범위가 안정적
            return _norm_text(cell) >= _norm_text(value)
        if op == "lte":
            return _norm_text(cell) <= _norm_text(value)
        return False

    if filters:
        try:
            for f in filters:
                if not isinstance(f, dict):
                    raise ValueError("filter must be object")
                history = [r for r in history if _check_one(r, f)]
        except ValueError as e:
            return [{"error": "bad_filter", "detail": str(e), "lot_id": lot_id}]

    if not history:
        return [{"error": "not_found", "lot_id": lot_id, "detail": "filtered_result_empty"}]

    if limit is not None:
        history = history[-limit:]

    return history
