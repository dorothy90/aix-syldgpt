"""
LOT 그래프 생성을 위한 유틸리티 함수들
"""

import os
import re
import sqlite3
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # GUI 백엔드 없이 사용
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv(override=True)


def extract_lot_id(question: str) -> Optional[str]:
    """
    질문에서 LOT ID를 추출합니다.

    예시:
        "lot1234 그려줘" -> "LOT1234"
        "LOT5678 그래프" -> "LOT5678"
    """
    # LOT 패턴 찾기 (대소문자 구분 없이)
    patterns = [
        r"\bLOT\s*(\w+)\b",
        r"\blot\s*(\w+)\b",
        r"\bLot\s*(\w+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            lot_id = match.group(1).upper()
            return f"LOT{lot_id}" if not lot_id.startswith("LOT") else lot_id

    return None


def get_db_connection():
    """
    데이터베이스 연결을 반환합니다.
    환경변수에서 DB 경로를 가져오거나 기본값 사용
    """
    db_path = os.getenv("LOT_DB_PATH", "03-Modules/03-Use-Cases/Chinook.db")

    # 절대 경로로 변환
    if not os.path.isabs(db_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        db_path = os.path.join(project_root, db_path)

    return sqlite3.connect(db_path)


def execute_lot_query(lot_id: str) -> Optional[pd.DataFrame]:
    """
    LOT ID를 기반으로 SQL 쿼리를 실행하고 결과를 반환합니다.

    실제 구현에서는 LOT ID에 맞는 쿼리를 동적으로 생성해야 합니다.
    여기서는 예시로 간단한 쿼리를 실행합니다.
    """
    try:
        conn = get_db_connection()

        # 예시 쿼리: LOT ID를 기반으로 데이터 조회
        # 실제로는 LOT 테이블 구조에 맞게 쿼리를 수정해야 합니다
        query = f"""
        SELECT * FROM Invoice
        WHERE InvoiceId <= 10
        ORDER BY InvoiceId
        LIMIT 20
        """

        # SQL 실행
        df = pd.read_sql_query(query, conn)
        conn.close()

        return df if not df.empty else None

    except Exception as e:
        print(f"SQL 쿼리 실행 오류: {e}")
        return None


def create_lot_plot(df: pd.DataFrame, lot_id: str) -> Optional[str]:
    """
    데이터프레임을 기반으로 그래프를 생성하고 Base64 인코딩된 이미지를 반환합니다.

    Args:
        df: 그래프를 그릴 데이터프레임
        lot_id: LOT ID (제목에 사용)

    Returns:
        Base64 인코딩된 PNG 이미지 문자열 (data:image/png;base64, 제외)
    """
    try:
        # 그래프 생성
        plt.figure(figsize=(10, 6))

        # 예시: 첫 번째 숫자 컬럼을 찾아서 그래프 그리기
        numeric_cols = df.select_dtypes(include=["number"]).columns

        if len(numeric_cols) > 0:
            # 첫 번째 숫자 컬럼을 y축으로 사용
            y_col = numeric_cols[0]
            x_col = df.index if "InvoiceId" not in df.columns else "InvoiceId"

            if isinstance(x_col, str) and x_col in df.columns:
                plt.plot(df[x_col], df[y_col], marker="o", linewidth=2, markersize=6)
                plt.xlabel(x_col)
            else:
                plt.plot(df[y_col], marker="o", linewidth=2, markersize=6)
                plt.xlabel("Index")

            plt.ylabel(y_col)
            plt.title(f"{lot_id} 추세 그래프", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        else:
            # 숫자 컬럼이 없으면 바 차트로 표시
            plt.bar(range(len(df)), [1] * len(df))
            plt.title(f"{lot_id} 데이터", fontsize=14, fontweight="bold")
            plt.xlabel("Index")
            plt.ylabel("Count")
            plt.tight_layout()

        # 이미지를 Base64로 변환
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return image_base64

    except Exception as e:
        print(f"그래프 생성 오류: {e}")
        plt.close()
        return None
