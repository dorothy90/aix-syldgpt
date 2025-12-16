from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

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


def get_lot_history(lot_id: str) -> List[Dict[str, Any]]:
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

    return sorted(history, key=_sort_key)
