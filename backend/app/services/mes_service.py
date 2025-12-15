from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import csv


@dataclass(frozen=True)
class MesResult:
    lot_id: str
    kind: str  # "lot_status" | "lot_history"
    data: dict


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def get_lot_status(lot_id: str) -> MesResult:
    """
    임시 구현: CSV에서 lot_status 정보를 조회
    """
    rows = _read_csv_rows(_DATA_DIR / "lot_status.csv")
    row = next((r for r in rows if (r.get("lot_id") or "").strip() == lot_id), None)
    if row is None:
        return MesResult(lot_id=lot_id, kind="lot_status", data={"found": False})
    return MesResult(lot_id=lot_id, kind="lot_status", data={"found": True, "row": row})


def get_lot_history(lot_id: str) -> MesResult:
    """
    임시 구현: CSV에서 lot_history 정보를 조회
    """
    rows = _read_csv_rows(_DATA_DIR / "lot_history.csv")
    history = [r for r in rows if (r.get("lot_id") or "").strip() == lot_id]
    if not history:
        return MesResult(lot_id=lot_id, kind="lot_history", data={"found": False, "events": []})
    # 시간순 정렬(있으면)
    def _sort_key(r: Dict[str, str]):
        return (r.get("event_time") or "").strip()

    history_sorted = sorted(history, key=_sort_key)
    return MesResult(
        lot_id=lot_id,
        kind="lot_history",
        data={"found": True, "events": history_sorted},
    )


