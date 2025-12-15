from fastapi import APIRouter, HTTPException

from app.services.mes_service import get_lot_history, get_lot_status

router = APIRouter(prefix="/api/mes", tags=["mes"])


@router.get("/lot_status/{lot_id}")
async def lot_status(lot_id: str):
    """
    임시 MES API: LOT 현재 상태 조회 (CSV 기반)
    """
    result = get_lot_status(lot_id.strip())
    return {"lot_id": result.lot_id, "kind": result.kind, "data": result.data}


@router.get("/lot_history/{lot_id}")
async def lot_history(lot_id: str):
    """
    임시 MES API: LOT 이력 조회 (CSV 기반)
    """
    result = get_lot_history(lot_id.strip())
    return {"lot_id": result.lot_id, "kind": result.kind, "data": result.data}


