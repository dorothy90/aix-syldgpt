"""
반도체 공정 데이터 생성 스크립트
- 실제 반도체 공정명으로 oper 변경
- 100만 줄의 시계열 데이터 생성
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 반도체 공정 이름 정의 (실제 반도체 FAB에서 사용하는 형태)
OPER_NAMES = [
    "GT PLUG ETCH CD",       # Gate Plug Etch Critical Dimension
    "BLC ETCH CD HV",        # Bitline Contact Etch CD High Voltage
    "STI CMP THK",           # Shallow Trench Isolation CMP Thickness
    "METAL CVD RATE",        # Metal Chemical Vapor Deposition Rate
    "GATE OX THK",           # Gate Oxide Thickness
    "SPACER DEP CD",         # Spacer Deposition Critical Dimension
    "ILD CMP UNIF",          # Inter Layer Dielectric CMP Uniformity
    "CONTACT ETCH DEPTH",    # Contact Etch Depth
    "PMD DEP RATE",          # Pre Metal Dielectric Deposition Rate
    "BARRIER ALD THK",       # Barrier Atomic Layer Deposition Thickness
    "W PLUG FILL",           # Tungsten Plug Fill
    "POLY ETCH CD",          # Polysilicon Etch Critical Dimension
    "HM STRIP RATE",         # Hard Mask Strip Rate
    "DIFF ANNEAL TEMP",      # Diffusion Anneal Temperature
    "ION IMP DOSE",          # Ion Implantation Dose
]

# 파라미터 정의 (반도체 공정에서 실제로 측정하는 값들)
PARAMETERS = {
    "GT PLUG ETCH CD": ["CD_TOP", "CD_BTM", "DEPTH", "TILT", "PRESSURE"],
    "BLC ETCH CD HV": ["CD_TOP", "CD_MID", "PROFILE", "VOLTAGE", "CURRENT"],
    "STI CMP THK": ["THK_CENTER", "THK_EDGE", "UNIFORMITY", "PRESSURE", "REMOVAL_RATE"],
    "METAL CVD RATE": ["DEP_RATE", "TEMP", "FLOW_RATE", "THK", "RESISTIVITY"],
    "GATE OX THK": ["THK", "UNIFORMITY", "TEMP", "O2_FLOW", "BREAKDOWN_V"],
    "SPACER DEP CD": ["CD", "THK", "PROFILE", "TEMP", "DEP_RATE"],
    "ILD CMP UNIF": ["UNIFORMITY", "THK_PRE", "THK_POST", "PRESSURE", "REMOVAL_RATE"],
    "CONTACT ETCH DEPTH": ["DEPTH", "CD", "PROFILE", "SELECTIVITY", "PRESSURE"],
    "PMD DEP RATE": ["DEP_RATE", "THK", "STRESS", "TEMP", "FLOW_RATE"],
    "BARRIER ALD THK": ["THK", "UNIFORMITY", "RESISTIVITY", "TEMP", "CYCLE_TIME"],
    "W PLUG FILL": ["FILL_RATE", "VOID_RATIO", "THK", "RESISTIVITY", "TEMP"],
    "POLY ETCH CD": ["CD", "PROFILE", "DEPTH", "SELECTIVITY", "PRESSURE"],
    "HM STRIP RATE": ["STRIP_RATE", "UNIFORMITY", "TEMP", "FLOW_RATE", "PRESSURE"],
    "DIFF ANNEAL TEMP": ["TEMP", "RAMP_RATE", "SOAK_TIME", "SHEET_RS", "UNIFORMITY"],
    "ION IMP DOSE": ["DOSE", "ENERGY", "TILT", "TWIST", "UNIFORMITY"],
}

# 파라미터별 기본값 범위 설정
PARAM_RANGES = {
    "CD_TOP": (45, 55),           # nm
    "CD_BTM": (40, 50),           # nm
    "CD_MID": (42, 52),           # nm
    "CD": (40, 60),               # nm
    "DEPTH": (100, 200),          # nm
    "THK": (10, 100),             # nm (또는 Angstrom)
    "THK_CENTER": (50, 70),       # nm
    "THK_EDGE": (45, 65),         # nm
    "THK_PRE": (500, 600),        # nm
    "THK_POST": (300, 400),       # nm
    "TILT": (0, 3),               # degree
    "TWIST": (0, 5),              # degree
    "PRESSURE": (1, 10),          # Torr
    "TEMP": (200, 800),           # Celsius
    "VOLTAGE": (100, 500),        # V
    "CURRENT": (0.5, 5),          # A
    "FLOW_RATE": (100, 500),      # sccm
    "O2_FLOW": (50, 200),         # sccm
    "DEP_RATE": (10, 100),        # nm/min
    "REMOVAL_RATE": (100, 500),   # nm/min
    "STRIP_RATE": (50, 200),      # nm/min
    "FILL_RATE": (50, 150),       # nm/min
    "UNIFORMITY": (95, 99.5),     # %
    "PROFILE": (85, 92),          # degree (이상적 90도)
    "SELECTIVITY": (10, 50),      # ratio
    "RESISTIVITY": (10, 50),      # uOhm-cm
    "SHEET_RS": (10, 100),        # Ohm/sq
    "BREAKDOWN_V": (5, 15),       # V
    "STRESS": (-500, 500),        # MPa
    "VOID_RATIO": (0, 5),         # %
    "CYCLE_TIME": (1, 10),        # sec
    "RAMP_RATE": (10, 50),        # C/sec
    "SOAK_TIME": (30, 120),       # sec
    "DOSE": (1e12, 1e16),         # atoms/cm2
    "ENERGY": (1, 100),           # keV
}


def generate_filter_data():
    """filter_data.csv 생성 - oper와 파라미터 매핑"""
    rows = []
    for oper, params in PARAMETERS.items():
        for param in params:
            rows.append({"oper": oper, "para": param})

    df = pd.DataFrame(rows)
    df.to_csv("supervisor/data/filter_data.csv", index=False)
    print(f"filter_data.csv 생성 완료: {len(df)} rows")
    return df


def generate_relation_data():
    """relation_data.csv 생성 - oper 간의 관계"""
    relations = [
        ("GT PLUG ETCH CD", ["BLC ETCH CD HV", "POLY ETCH CD"], ["CD_TOP", "CD_BTM", "DEPTH"]),
        ("BLC ETCH CD HV", ["GT PLUG ETCH CD", "CONTACT ETCH DEPTH"], ["CD_TOP", "VOLTAGE", "CURRENT"]),
        ("STI CMP THK", ["ILD CMP UNIF", "GATE OX THK"], ["THK_CENTER", "UNIFORMITY", "REMOVAL_RATE"]),
        ("METAL CVD RATE", ["BARRIER ALD THK", "W PLUG FILL"], ["DEP_RATE", "THK", "RESISTIVITY"]),
        ("GATE OX THK", ["STI CMP THK", "DIFF ANNEAL TEMP"], ["THK", "TEMP", "UNIFORMITY"]),
        ("SPACER DEP CD", ["POLY ETCH CD", "GATE OX THK"], ["CD", "THK", "PROFILE"]),
        ("ILD CMP UNIF", ["STI CMP THK", "PMD DEP RATE"], ["UNIFORMITY", "REMOVAL_RATE", "PRESSURE"]),
        ("CONTACT ETCH DEPTH", ["BLC ETCH CD HV", "W PLUG FILL"], ["DEPTH", "CD", "SELECTIVITY"]),
        ("PMD DEP RATE", ["ILD CMP UNIF", "METAL CVD RATE"], ["DEP_RATE", "THK", "STRESS"]),
        ("BARRIER ALD THK", ["METAL CVD RATE", "W PLUG FILL"], ["THK", "UNIFORMITY", "RESISTIVITY"]),
        ("W PLUG FILL", ["BARRIER ALD THK", "CONTACT ETCH DEPTH"], ["FILL_RATE", "VOID_RATIO", "RESISTIVITY"]),
        ("POLY ETCH CD", ["GT PLUG ETCH CD", "SPACER DEP CD"], ["CD", "PROFILE", "DEPTH"]),
        ("HM STRIP RATE", ["POLY ETCH CD", "GT PLUG ETCH CD"], ["STRIP_RATE", "UNIFORMITY", "TEMP"]),
        ("DIFF ANNEAL TEMP", ["GATE OX THK", "ION IMP DOSE"], ["TEMP", "SHEET_RS", "UNIFORMITY"]),
        ("ION IMP DOSE", ["DIFF ANNEAL TEMP", "POLY ETCH CD"], ["DOSE", "ENERGY", "UNIFORMITY"]),
    ]

    rows = []
    for oper, related_opers, related_params in relations:
        rows.append({
            "oper": oper,
            "related_oper": ",".join(related_opers),
            "related_para": ",".join(related_params)
        })

    df = pd.DataFrame(rows)
    df.to_csv("supervisor/data/relation_data.csv", index=False)
    print(f"relation_data.csv 생성 완료: {len(df)} rows")
    return df


def generate_timeseries_data(total_rows=1_000_000):
    """timeseries_data.csv 생성 - 100만 줄의 시계열 데이터"""
    print(f"시계열 데이터 생성 중... (목표: {total_rows:,} rows)")

    # 시작 시간 설정
    start_date = datetime(2024, 1, 1)

    # 각 oper-para 조합별로 필요한 데이터 포인트 수 계산
    oper_para_combinations = []
    for oper, params in PARAMETERS.items():
        for param in params:
            oper_para_combinations.append((oper, param))

    total_combinations = len(oper_para_combinations)
    rows_per_combination = total_rows // total_combinations
    extra_rows = total_rows % total_combinations

    print(f"총 oper-para 조합 수: {total_combinations}")
    print(f"조합당 데이터 포인트: {rows_per_combination:,}")

    # 데이터 생성 (청크 단위로 처리)
    chunk_size = 100_000
    all_data = []
    current_row = 0

    for idx, (oper, param) in enumerate(oper_para_combinations):
        # 각 조합에 할당할 행 수
        rows_for_this = rows_per_combination + (1 if idx < extra_rows else 0)

        # 파라미터 범위 가져오기
        param_range = PARAM_RANGES.get(param, (0, 100))
        base_value = (param_range[0] + param_range[1]) / 2
        value_range = (param_range[1] - param_range[0]) / 2

        # 시간 간격 계산 (1분~6시간 사이 랜덤)
        time_intervals = np.random.randint(1, 360, size=rows_for_this)  # 분 단위

        # 시계열 값 생성 (트렌드 + 노이즈 + 주기성)
        t = np.arange(rows_for_this)

        # 기본 트렌드 (약간의 drift)
        trend = np.cumsum(np.random.randn(rows_for_this) * 0.01)

        # 주기적 패턴 (일별, 주별)
        daily_pattern = np.sin(2 * np.pi * t / (24 * 60 / 6)) * value_range * 0.1
        weekly_pattern = np.sin(2 * np.pi * t / (7 * 24 * 60 / 6)) * value_range * 0.05

        # 노이즈
        noise = np.random.randn(rows_for_this) * value_range * 0.15

        # 최종 값 계산
        values = base_value + trend + daily_pattern + weekly_pattern + noise

        # 범위 내로 클리핑
        values = np.clip(values, param_range[0], param_range[1])

        # 특수 처리 (DOSE는 지수 스케일)
        if param == "DOSE":
            values = 10 ** np.random.uniform(12, 16, size=rows_for_this)

        # 시간 생성
        timestamps = []
        current_time = start_date + timedelta(days=idx * 30)  # 각 조합별로 다른 시작점
        for interval in time_intervals:
            timestamps.append(current_time)
            current_time += timedelta(minutes=int(interval))

        # 데이터 추가
        for i in range(rows_for_this):
            all_data.append({
                "oper": oper,
                "para": param,
                "end_tm": timestamps[i].strftime("%Y-%m-%d %H:%M:%S"),
                "value": round(values[i], 4)
            })
            current_row += 1

            # 진행 상황 출력
            if current_row % 100_000 == 0:
                print(f"  진행: {current_row:,} / {total_rows:,} ({current_row/total_rows*100:.1f}%)")

    # DataFrame 생성 및 저장
    print("DataFrame 생성 중...")
    df = pd.DataFrame(all_data)

    # 시간순 정렬 (선택사항)
    # df = df.sort_values(by='end_tm').reset_index(drop=True)

    print("CSV 파일 저장 중...")
    df.to_csv("supervisor/data/timeseries_data.csv", index=False)
    print(f"timeseries_data.csv 생성 완료: {len(df):,} rows")

    return df


def main():
    print("=" * 60)
    print("반도체 공정 데이터 생성 시작")
    print("=" * 60)

    # 1. filter_data.csv 생성
    print("\n[1/3] filter_data.csv 생성")
    generate_filter_data()

    # 2. relation_data.csv 생성
    print("\n[2/3] relation_data.csv 생성")
    generate_relation_data()

    # 3. timeseries_data.csv 생성 (100만 줄)
    print("\n[3/3] timeseries_data.csv 생성 (100만 줄)")
    generate_timeseries_data(total_rows=1_000_000)

    print("\n" + "=" * 60)
    print("모든 데이터 생성 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

