"""CSV 필터링 및 차트 생성 도구"""

import os
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # GUI 없이 이미지 저장 (멀티스레드 안전)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 데이터 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_filter_csv(
    oper: Optional[str] = None,
    para: Optional[str] = None,
) -> str:
    """filter_data.csv를 로드하고 조건에 맞게 필터링

    Args:
        oper: 필터링할 oper 값 (예: OP001)
        para: 필터링할 para 값 (예: TEMP)

    Returns:
        필터링된 데이터의 문자열 표현
    """
    # #region agent log
    import json

    open(
        "/Users/daehwankim/Documents/langgraph-tutorial-main/.cursor/debug.log", "a"
    ).write(
        json.dumps(
            {
                "location": "tools.py:load_and_filter_csv",
                "message": "load_and_filter_csv CALLED",
                "data": {"oper": oper, "para": para},
                "hypothesisId": "A,B",
                "timestamp": __import__("time").time(),
            }
        )
        + "\n"
    )
    # #endregion
    csv_path = DATA_DIR / "filter_data.csv"
    df = pd.read_csv(csv_path)

    if oper:
        df = df[df["oper"] == oper]
    if para:
        df = df[df["para"] == para]

    if df.empty:
        return f"조건에 맞는 데이터가 없습니다. (oper={oper}, para={para})"

    result = f"총 {len(df)}개의 데이터를 찾았습니다.\n\n"
    result += df.to_string(index=False)
    return result


def create_chart(
    oper: str,
    para: str,
    chart_type: str = "line",
) -> str:
    """timeseries_data.csv에서 데이터를 추출하여 시계열 차트 생성

    Args:
        oper: 조회할 oper 값 (예: OP001)
        para: 조회할 para 값 (예: TEMP)
        chart_type: 차트 유형 (line, bar)

    Returns:
        생성된 차트 파일 경로 또는 오류 메시지
    """
    # #region agent log
    import json

    open(
        "/Users/daehwankim/Documents/langgraph-tutorial-main/.cursor/debug.log", "a"
    ).write(
        json.dumps(
            {
                "location": "tools.py:create_chart",
                "message": "create_chart CALLED",
                "data": {"oper": oper, "para": para, "chart_type": chart_type},
                "hypothesisId": "E",
                "timestamp": __import__("time").time(),
            }
        )
        + "\n"
    )
    # #endregion
    csv_path = DATA_DIR / "timeseries_data.csv"
    df = pd.read_csv(csv_path)
    # #region agent log
    import json

    open(
        "/Users/daehwankim/Documents/langgraph-tutorial-main/.cursor/debug.log", "a"
    ).write(
        json.dumps(
            {
                "location": "tools.py:create_chart:after_read",
                "message": "CSV loaded",
                "data": {"total_rows": len(df), "csv_path": str(csv_path)},
                "hypothesisId": "D",
                "timestamp": __import__("time").time(),
            }
        )
        + "\n"
    )
    # #endregion

    # 필터링
    filtered = df[(df["oper"] == oper) & (df["para"] == para)]
    # #region agent log
    import json

    open(
        "/Users/daehwankim/Documents/langgraph-tutorial-main/.cursor/debug.log", "a"
    ).write(
        json.dumps(
            {
                "location": "tools.py:create_chart:after_filter",
                "message": "Filtered result",
                "data": {"filtered_count": len(filtered), "oper": oper, "para": para},
                "hypothesisId": "C",
                "timestamp": __import__("time").time(),
            }
        )
        + "\n"
    )
    # #endregion

    if filtered.empty:
        return f"조건에 맞는 데이터가 없습니다. (oper={oper}, para={para})"

    # 날짜 변환
    filtered = filtered.copy()
    filtered["end_tm"] = pd.to_datetime(filtered["end_tm"])
    filtered = filtered.sort_values("end_tm")

    # #region agent log
    import json

    open(
        "/Users/daehwankim/Documents/langgraph-tutorial-main/.cursor/debug.log", "a"
    ).write(
        json.dumps(
            {
                "location": "tools.py:create_chart:before_plot",
                "message": "Data before plotting",
                "data": {
                    "value_min": float(filtered["value"].min()),
                    "value_max": float(filtered["value"].max()),
                    "value_sample": filtered["value"].head(3).tolist(),
                    "date_sample": filtered["end_tm"].head(3).astype(str).tolist(),
                },
                "hypothesisId": "C",
                "timestamp": __import__("time").time(),
            }
        )
        + "\n"
    )
    # #endregion

    # 차트 생성 (객체 지향 API 사용 - 멀티스레드 안전)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig = Figure(figsize=(12, 6), facecolor="white")
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    if chart_type == "bar":
        ax.bar(filtered["end_tm"], filtered["value"], width=0.02, alpha=0.7)
    else:  # line (기본)
        ax.plot(
            filtered["end_tm"],
            filtered["value"],
            marker="o",
            markersize=3,
            linewidth=1,
            color="blue",
        )

    ax.set_xlabel("시간")
    ax.set_ylabel(para)
    ax.set_title(f"{oper} - {para} 시계열 그래프")
    ax.grid(True, alpha=0.3)

    # X축 날짜 포맷
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    fig.tight_layout()

    # 저장
    filename = f"{oper}_{para}_{chart_type}.png"
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=100, facecolor="white")

    # #region agent log
    import json
    import os

    file_exists = output_path.exists()
    file_size = os.path.getsize(output_path) if file_exists else 0
    open(
        "/Users/daehwankim/Documents/langgraph-tutorial-main/.cursor/debug.log", "a"
    ).write(
        json.dumps(
            {
                "location": "tools.py:create_chart:after_save",
                "message": "Chart saved",
                "data": {
                    "output_path": str(output_path),
                    "file_exists": file_exists,
                    "file_size_bytes": file_size,
                },
                "hypothesisId": "D",
                "timestamp": __import__("time").time(),
            }
        )
        + "\n"
    )
    # #endregion

    return f"차트가 생성되었습니다: {output_path}"


def get_related_info(oper: str) -> str:
    """relation_data.csv에서 특정 oper의 연관 oper와 para 목록을 조회

    Args:
        oper: 조회할 oper 값 (예: OP001)

    Returns:
        연관된 oper와 para 목록의 문자열 표현
    """
    # #region agent log
    import json

    open(
        "/Users/daehwankim/Documents/langgraph-tutorial-main/.cursor/debug.log", "a"
    ).write(
        json.dumps(
            {
                "location": "tools.py:get_related_info",
                "message": "get_related_info CALLED",
                "data": {"oper": oper},
                "hypothesisId": "A",
                "timestamp": __import__("time").time(),
            }
        )
        + "\n"
    )
    # #endregion
    csv_path = DATA_DIR / "relation_data.csv"

    if not csv_path.exists():
        return f"relation_data.csv 파일이 존재하지 않습니다."

    df = pd.read_csv(csv_path)

    # 해당 oper 찾기
    row = df[df["oper"] == oper]

    if row.empty:
        return f"'{oper}'에 대한 연관 정보가 없습니다."

    row = row.iloc[0]
    related_opers = row["related_oper"]
    related_paras = row["related_para"]

    result = f"📋 {oper}의 연관 정보\n"
    result += f"{'='*40}\n\n"
    result += f"🔗 연관 Oper: {related_opers}\n"
    result += f"📊 연관 Para: {related_paras}\n"

    # 리스트로 파싱
    oper_list = [o.strip() for o in related_opers.split(",")]
    para_list = [p.strip() for p in related_paras.split(",")]

    result += f"\n📌 상세 목록:\n"
    result += f"  - 연관 Oper ({len(oper_list)}개): {', '.join(oper_list)}\n"
    result += f"  - 연관 Para ({len(para_list)}개): {', '.join(para_list)}\n"

    # #region agent log
    import json

    open(
        "/Users/daehwankim/Documents/langgraph-tutorial-main/.cursor/debug.log", "a"
    ).write(
        json.dumps(
            {
                "location": "tools.py:get_related_info:result",
                "message": "Related info found",
                "data": {
                    "oper": oper,
                    "related_opers": oper_list,
                    "related_paras": para_list,
                },
                "hypothesisId": "A",
                "timestamp": __import__("time").time(),
            }
        )
        + "\n"
    )
    # #endregion

    return result
