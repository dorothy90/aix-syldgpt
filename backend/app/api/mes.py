from fastapi import APIRouter
from pydantic import BaseModel

from app.services.mes_service import get_lot_history, get_lot_status

router = APIRouter(prefix="/api/mes", tags=["mes"])


class LotIdRequest(BaseModel):
    lot_id: str


@router.post("/lot_status")
async def post_lot_status(req: LotIdRequest):
    """
    MES API: LOT 현재 상태 조회 (Excel 기반)
    - 항상 list[dict] 형태의 순수 JSON 배열을 반환
    """
    return get_lot_status(req.lot_id)


@router.post("/lot_history")
async def post_lot_history(req: LotIdRequest):
    """
    MES API: LOT 이력 조회 (Excel 기반)
    - 항상 list[dict] 형태의 순수 JSON 배열을 반환
    """
    return get_lot_history(req.lot_id)


@router.get("/lot_status/{lot_id}")
async def lot_status(lot_id: str):
    """
    임시 MES API: LOT 현재 상태 조회 (CSV 기반)
    """
    return [
        {
            "error": "deprecated",
            "detail": "Use POST /api/mes/lot_status with JSON body {lot_id}",
        }
    ]


@router.get("/lot_history/{lot_id}")
async def lot_history(lot_id: str):
    """
    임시 MES API: LOT 이력 조회 (CSV 기반)
    """
    return [
        {
            "error": "deprecated",
            "detail": "Use POST /api/mes/lot_history with JSON body {lot_id}",
        }
    ]
