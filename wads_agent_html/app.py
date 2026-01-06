import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random
import webbrowser
import tempfile

def load_html_to_dataframe(folder_path):
    """
    HTML 파일들을 lotcd, end_tm, ctn_desc, html 구조의 DataFrame으로 변환
    - lotcd: 6E2, 4SA, 5NA 중 랜덤
    - end_tm: 1월 1일~5일, 하루 2개씩 랜덤
    """
    html_files = sorted(Path(folder_path).glob("*.html"))

    # lotcd 후보
    lotcd_list = ["6E2", "4SA", "5NA"]

    # end_tm 생성: 1월 1일~5일, 하루 2개씩 (총 10개 슬롯)
    base_date = datetime(2026, 1, 1)
    end_tm_slots = []
    for day_offset in range(5):  # 1월 1일 ~ 5일
        date = base_date + timedelta(days=day_offset)
        # 하루에 2개씩 랜덤 시간 생성
        for _ in range(2):
            hour = random.randint(8, 22)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            end_tm_slots.append(date.replace(hour=hour, minute=minute, second=second))

    # 시간순 정렬
    end_tm_slots.sort()

    data = []
    for idx, file_path in enumerate(html_files):
        # lotcd: 랜덤 선택
        lotcd = random.choice(lotcd_list)

        # end_tm: 미리 생성한 슬롯에서 순서대로 (파일 개수만큼만 사용)
        end_tm = end_tm_slots[idx] if idx < len(end_tm_slots) else end_tm_slots[-1]
        end_tm_str = end_tm.strftime("%Y-%m-%d %H:%M:%S")

        # ctn_desc: step01, step02, ...
        ctn_desc = f"step{idx + 1:02d}"

        # HTML 내용 읽기
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        data.append({
            "lotcd": lotcd,
            "end_tm": end_tm_str,
            "ctn_desc": ctn_desc,
            "html": html_content
        })

    df = pd.DataFrame(data)
    return df


def filter_and_display(df, lotcd=None, ctn_desc=None, end_tm_start=None, end_tm_end=None):
    """
    조건에 맞는 row를 필터링하고 HTML을 브라우저에서 표시
    """
    filtered = df.copy()

    if lotcd:
        filtered = filtered[filtered["lotcd"] == lotcd]
    if ctn_desc:
        filtered = filtered[filtered["ctn_desc"] == ctn_desc]
    if end_tm_start:
        filtered = filtered[filtered["end_tm"] >= end_tm_start]
    if end_tm_end:
        filtered = filtered[filtered["end_tm"] <= end_tm_end]

    if filtered.empty:
        print("조건에 맞는 데이터가 없습니다.")
        return None

    html_content = filtered.iloc[0]["html"]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        temp_path = f.name

    webbrowser.open(f"file://{temp_path}")
    print(f"표시됨: lotcd={filtered.iloc[0]['lotcd']}, ctn_desc={filtered.iloc[0]['ctn_desc']}, end_tm={filtered.iloc[0]['end_tm']}")

    return filtered


if __name__ == "__main__":
    # 재현 가능하게 시드 설정 (원하면 제거)
    random.seed(42)

    folder_path = "/Users/daehwankim/Documents/langgraph-tutorial-main/wads_agent_html"

    df = load_html_to_dataframe(folder_path)
    df.to_csv("wads_agent_html.csv", index=False)

    print("=== DataFrame 구조 ===")
    print(df[["lotcd", "end_tm", "ctn_desc"]].to_string())
    print(f"\n총 {len(df)}개 row 생성됨")

    # 필터링 예시
    # filter_and_display(df, lotcd="6E2")
    # filter_and_display(df, end_tm_start="2026-01-03 00:00:00", end_tm_end="2026-01-03 23:59:59")
