#!/usr/bin/env python3
"""
build_tcn_dataset.py - TCN 학습용 데이터셋 생성 (최종공통엔진스펙 §3,§5,§6,§9 준수)

파이프라인:
    시나리오 인원표 -> (1분 웨이포인트, 실좌표) -> ns-3 dataset 모드
      -> 분당 셀별 관측: ue_count(접속 UE), isac_count, mean_sinr_db, ho_in, ho_out,
         traffic_demand_mbps(UE별 확률적 수요 합, 유형별 선행시차 §6), presence_count
    -> Python 파생특징:
      Isac_density(σ8%), Prb_utilization(D/C), Neighbor_mean_load(인접셀 PRB), Hour_sin/cos
    -> 다일 결합 + 슬라이딩 윈도우 [F,60] -> [15] (.npz)

시간축(§3): 09:00~21:00, 1분 관측 -> 720스텝/일. 입력 L=60, 예측 H=15.
안테나: 고정 16 (baseline). RL이 이 기준선을 개선(§2). 요구 트래픽 D는 안테나와
무관(§4)하지만, 입력 특징 중 PRB/SINR/HO는 이 기준 안테나 하의 관측값이다.

사용법 (ns-3 루트에서):
    python3 build_tcn_dataset.py                       # 일반일과, 1일, 풀스케일
    python3 build_tcn_dataset.py --days 15
    python3 build_tcn_dataset.py --scenarios 일반일과 --scale 0.3   # 빠른 검증
    # 제공된 TCN의 학습 데이터 (약 8분 반)
    python3 build_tcn_dataset.py --scenarios 일반일과 시험기간 축제 --days 31 --plot --outdir tcn-dataset-v3-5g-31d
    # 최소 실행 (약 1분): QUICK=1 run_tcn_v3.sh용 3일 (학습 0일 / 검증 1일 / 테스트 2일)
    python3 build_tcn_dataset.py --scenarios 일반일과 시험기간 축제 --days 3 --outdir tcn-dataset-quick

--days 1이면 날 변동 없이 시나리오 인원표 그대로 하루를 만든다. --days 2 이상이면 날마다
변동을 주고, 앞쪽 날은 --days와 무관하게 같다 (--days 3의 0~2일 = --days 31의 0~2일).
"""

import argparse
import csv
import math
import subprocess
import sys
import zlib
from pathlib import Path

import run_campaign as rc

# ---------------------------------------------------------------------------
# 스펙 파라미터
# ---------------------------------------------------------------------------
MINUTES_PER_DAY = 720          # 09:00~21:00, 1분 스텝 (§3)
INPUT_L = 60                   # 입력 윈도우(분) (§3,§9)
HORIZON_H = 15                 # 예측 지평(분) (§3,§9)
BASELINE_ANTENNAS = 16         # 고정 baseline (§2)

# 활성안테나 16 기준 셀 용량 C_max (§2, C = C_max x A/16). U=min(1,D/C).
# 5G NR n78: 100 MHz x TDD(DDDSU) 하향 비율 0.74 x ITU-R M.2410 Dense Urban DL 평균
# 스펙트럼 효율 7.8 b/s/Hz = 577 Mbps.
CELL_CMAX_MBPS = 580.0
# 1인당 수요는 C_max와 같은 배수로 키워 사용률 U=D/C를 5G 이전 설계(C_max 200 Mbps)와
# 같게 유지한다.
DEMAND_SCALE = CELL_CMAX_MBPS / 200.0

# 건물 유형별 1인당 요구 트래픽[Mbps]과 인원->트래픽 선행 시차[분] (§6).
# "인원 != 트래픽": 유형마다 1인당 부하·선행 시차가 다르다.
TYPE_RATE_MBPS = {k: v * DEMAND_SCALE
                  for k, v in {"강의동": 0.8, "도서관": 0.3, "기숙사": 0.6, "이벤트": 0.5}.items()}
TYPE_LAG_MIN = {"강의동": 5, "도서관": 2, "기숙사": 0, "이벤트": 12}
DORM_EVENING_MULT = 2.0        # 기숙사 저녁(>=18시) 스트리밍 배수 (§6,§7)

ISAC_SIGMA = 0.08              # ISAC 밀집도 측정오차 σ=8% (§5)
N_NEIGHBORS = 3                # Neighbor_mean_load 계산에 쓸 인접 셀 수 (§5)

# TCN 입력 특징 (§5, F=9; 시각은 sin/cos 2채널 -> 실제 10채널). 타깃은 첫 번째.
FEATURES = [
    "Traffic_demand_mbps",  # 1 (타깃)
    "Ue_count",             # 2
    "Isac_density",         # 3
    "Prb_utilization",      # 4
    "Mean_sinr_db",         # 5
    "Handover_in",          # 6
    "Handover_out",         # 7
    "Neighbor_mean_load",   # 8
    "Hour_sin",             # 9a
    "Hour_cos",             # 9b
]
TARGET = "Traffic_demand_mbps"


def cell_meta():
    """셀 id(1..N) -> (건물명, 유형, x, y). C++와 동일하게 (y, x) 정렬로 번호 부여."""
    order = sorted(range(len(rc.SITES)), key=lambda i: (rc.SITES[i][4], rc.SITES[i][3]))
    meta = {}
    for rank, i in enumerate(order):
        name, typ, gnb, x, y = rc.SITES[i]
        meta[rank + 1] = {"name": name, "type": typ, "x": x, "y": y}
    return meta


def neighbor_lists(meta, k=N_NEIGHBORS):
    """각 셀의 k개 최근접 인접 셀 id 목록."""
    cells = sorted(meta.keys())
    nbrs = {}
    for c in cells:
        cx, cy = meta[c]["x"], meta[c]["y"]
        d = sorted(
            (o for o in cells if o != c),
            key=lambda o: (meta[o]["x"] - cx) ** 2 + (meta[o]["y"] - cy) ** 2,
        )
        nbrs[c] = d[:k]
    return nbrs


def perturb_scenario(scenario_rows, rng, day_var):
    """하루치 인원표에 일간 변동(건물별·슬롯별 곱셈 노이즈)을 준다. 캠퍼스밖 인원은
    슬롯 합이 1000이 되도록 재계산(이동 보존, §11)."""
    n_slots = len(rc.SLOT_HOURS)
    new_rows = []
    for (name, typ, cnts) in scenario_rows:
        nc = [max(0, int(round(c * (1.0 + rng.gauss(0.0, day_var))))) for c in cnts]
        new_rows.append((name, typ, nc))
    new_off = []
    for t in range(n_slots):
        bsum = sum(r[2][t] for r in new_rows)
        new_off.append(max(0, 1000 - bsum))
    return new_rows, new_off


def run_dataset_sim(waypoints, sites_file, antenna_file, rates_file, minutes, rng_run, tag,
                    outdir, v2):
    """ns-3 dataset 모드를 실행하고 분당 셀별 관측 CSV(<tag>-obs.csv) 경로를 돌려준다.
    v2 = per-UE 확률적 생성 파라미터(연결지연·activity·수요 폭·P_D)."""
    prog_args = (
        f"{rc.NS3_PROGRAM} --dataset --datasetMinutes={minutes}"
        f" --waypoints={waypoints} --sitesFile={sites_file} --antennaFile={antenna_file}"
        f" --RngRun={rng_run} --simTag={tag} --outputDir={outdir}"
        f" --ratesFile={rates_file}"
        f" --v2ConnDelayMean={v2['conn_delay']}"
        f" --v2ActivityProb={v2['activity']}"
        f" --v2RateCv={v2['rate_cv']}"
        f" --v2Pd={v2['pd']}"
    )
    subprocess.run(
        [rc.NS3, "run", "--no-build", prog_args],
        cwd=rc.REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return outdir / f"{tag}-obs.csv"


def read_obs(path, n_cells, n_min):
    """<tag>-obs.csv -> obs[cell][minute] = dict(ue_count, isac_count, sinr, ho_in, ho_out,
    traffic, presence). minutes 0..n_min-1."""
    obs = {c: [None] * n_min for c in range(1, n_cells + 1)}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            c = int(r["cell"])
            m = int(r["minute"])
            if c in obs and 0 <= m < n_min:
                obs[c][m] = {
                    "ue_count": int(r["ue_count"]),      # 접속 UE 수
                    "isac_count": int(r["isac_count"]),
                    "sinr": float(r["mean_sinr_db"]),
                    "ho_in": int(r["ho_in"]),
                    "ho_out": int(r["ho_out"]),
                    # ns-3가 계산한 확률적 트래픽 + presence(연결 전 포함) 수
                    "traffic": float(r["traffic_demand_mbps"]),
                    "presence": int(r["presence_count"]),
                }
    return obs


def hour_of(minute):
    """분(0..720) -> 하루 시각(9.0 .. 21.0)."""
    return 9.0 + minute / 60.0


def derive_features(obs, meta, nbrs, rng):
    """분당 셀별 관측(obs)에 스펙 파생특징을 더해 rec[cell][minute] = {특징...} 생성.
    트래픽은 ns-3가 계산한 per-UE 확률적 합(obs['traffic'])을 그대로 쓴다."""
    n_min = len(next(iter(obs.values())))
    cells = sorted(meta.keys())

    # 1) 셀별 요구 트래픽 + PRB.
    traffic = {c: [0.0] * n_min for c in cells}
    prb = {c: [0.0] * n_min for c in cells}
    for c in cells:
        for m in range(n_min):
            o = obs[c][m]
            d = o["traffic"] if o else 0.0
            traffic[c][m] = d
            prb[c][m] = min(1.0, d / CELL_CMAX_MBPS) if CELL_CMAX_MBPS > 0 else 0.0

    # 2) 인접 셀 평균 PRB.
    rec = {c: [None] * n_min for c in cells}
    for c in cells:
        for m in range(n_min):
            o = obs[c][m] or {"ue_count": 0, "isac_count": 0, "sinr": 0.0, "ho_in": 0, "ho_out": 0}
            nb = nbrs[c]
            nload = (sum(prb[j][m] for j in nb) / len(nb)) if nb else 0.0
            isac_d = max(0.0, o["isac_count"] * (1.0 + rng.gauss(0.0, ISAC_SIGMA)))
            h = hour_of(m)
            rec[c][m] = {
                "Traffic_demand_mbps": round(traffic[c][m], 4),
                "Ue_count": o["ue_count"],
                "Isac_density": round(isac_d, 4),
                "Prb_utilization": round(prb[c][m], 4),
                "Mean_sinr_db": round(o["sinr"], 3),
                "Handover_in": o["ho_in"],
                "Handover_out": o["ho_out"],
                "Neighbor_mean_load": round(nload, 4),
                "Hour_sin": round(math.sin(2 * math.pi * h / 24.0), 5),
                "Hour_cos": round(math.cos(2 * math.pi * h / 24.0), 5),
            }
    return rec


def build_windows(records, feature_names, l=INPUT_L, h=HORIZON_H, target=TARGET):
    """records: list of dict(scenario, day, cell, rec[minute]->features). 슬라이딩
    윈도우 X[N,C,L], Y[N,H] 와 메타(scenario,day,cell,t)를 만든다 (하루 경계 내)."""
    import numpy as np

    ti = feature_names.index(target)
    xs, ys, meta_rows = [], [], []
    for item in records:
        rec = item["rec"]
        n_min = len(rec)
        arr = np.array(
            [[rec[m][f] for f in feature_names] for m in range(n_min)], dtype=np.float32
        )  # [n_min, C]
        # t = 예측 기준 시점(현재). 입력 = [t-L+1 .. t], 타깃 = [t+1 .. t+H].
        for t in range(l - 1, n_min - h):
            xs.append(arr[t - l + 1 : t + 1, :].T)  # [C, L]
            ys.append(arr[t + 1 : t + 1 + h, ti])   # [H]
            meta_rows.append((item["scenario"], item["day"], item["cell"], t))
    X = np.stack(xs).astype(np.float32) if xs else np.zeros((0, len(feature_names), l), np.float32)
    Y = np.stack(ys).astype(np.float32) if ys else np.zeros((0, h), np.float32)
    return X, Y, meta_rows


def _hhmm(minute):
    """분(0..720) -> 'HH:MM' 시각 라벨."""
    h = 9.0 + minute / 60.0
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        hh += 1
        mm = 0
    return f"{hh:02d}:{mm:02d}"


def plot_dataset(perminute_path, npz_path, outdir):
    """dataset_perminute.csv(+npz)로 데이터셋을 눈으로 확인하는 PNG들을 만든다.
    (1) 유형별 대표 셀의 인원->트래픽(선행시차), (2) 셀x시간 트래픽 히트맵,
    (3) 대표 셀의 전체 9특징, (4) TCN 표본 윈도우(입력60+예측15)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot] matplotlib/numpy 없음 -> 그래프 생략")
        return []

    rows = list(csv.DictReader(open(perminute_path)))
    # 첫 시나리오/첫 날만 그린다.
    scen = rows[0]["scenario"]
    day = rows[0]["day"]
    rows = [r for r in rows if r["scenario"] == scen and r["day"] == day]
    cells = sorted({int(r["cell"]) for r in rows})
    n_min = max(int(r["minute"]) for r in rows) + 1
    meta = cell_meta()

    def series(cell, field):
        s = [0.0] * n_min
        for r in rows:
            if int(r["cell"]) == cell:
                s[int(r["minute"])] = float(r[field])
        return s

    mins = list(range(n_min))
    xticks = list(range(0, n_min, 90))
    xlabels = [_hhmm(m) for m in xticks]
    outs = []

    # 대표 셀: 유형별 최대 피크 트래픽 셀.
    peak = {c: max(series(c, "Traffic_demand_mbps")) for c in cells}
    rep = {}
    for c in cells:
        t = meta[c]["type"]
        if t not in rep or peak[c] > peak[rep[t]]:
            rep[t] = c

    # (1) 유형별 대표 셀: Ue_count / Isac_density / Traffic (선행시차 확인).
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, (t, c) in zip(axes.ravel(), sorted(rep.items())):
        ax.plot(mins, series(c, "Ue_count"), color="tab:blue", lw=1.2, label="Ue_count")
        ax.plot(mins, series(c, "Isac_density"), color="tab:green", lw=0.9,
                alpha=0.7, label="Isac_density")
        ax2 = ax.twinx()
        ax2.plot(mins, series(c, "Traffic_demand_mbps"), color="tab:red", lw=1.4,
                 label="Traffic (Mbps)")
        ax.set_title(f"cell {c} - {rc.TYPE_EN.get(t, t)} ({meta[c]['name']})")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, fontsize=7)
        ax.set_ylabel("people")
        ax2.set_ylabel("Mbps", color="tab:red")
        ax.grid(True, alpha=0.25)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="upper right")
    fig.suptitle(f"[{scen}] Per-type representative cells: occupancy leads traffic")
    fig.tight_layout()
    p = outdir / "plot_representative_cells.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    outs.append(p)

    # (2) 셀 x 시간 트래픽 히트맵.
    grid = np.array([series(c, "Traffic_demand_mbps") for c in cells])
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, n_min, 0.5, len(cells) + 0.5])
    ax.set_yticks(cells)
    ax.set_yticklabels([f"c{c} {rc.TYPE_EN.get(meta[c]['type'], '')[:3]}" for c in cells],
                       fontsize=7)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, fontsize=8)
    ax.set_title(f"[{scen}] Traffic demand [Mbps]  (cell x time)")
    fig.colorbar(im, ax=ax, label="Mbps")
    fig.tight_layout()
    p = outdir / "plot_traffic_heatmap.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    outs.append(p)

    # (3) 대표 강의동 셀의 전체 특징 (TCN이 보는 입력).
    c = rep.get("강의동", cells[0])
    feats = [f for f in FEATURES if f not in ("Hour_sin", "Hour_cos")]
    fig, axes = plt.subplots(len(feats), 1, figsize=(12, 12), sharex=True)
    for ax, f in zip(axes, feats):
        ax.plot(mins, series(c, f), color="tab:purple", lw=1.0)
        ax.set_ylabel(f, fontsize=7)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xlabels, rotation=45, fontsize=8)
    fig.suptitle(f"[{scen}] cell {c} ({meta[c]['name']}) - all TCN input features")
    fig.tight_layout()
    p = outdir / "plot_all_features.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    outs.append(p)

    # (4) TCN 표본 윈도우: 입력 60분 + 예측 15분.
    if npz_path and Path(npz_path).exists():
        d = np.load(npz_path, allow_pickle=True)
        X, Y, mrows = d["X"], d["Y"], d["meta"]
        ti = list(d["feature_names"]).index(TARGET)
        # 오전 상승 구간의 강의동 셀 윈도우를 하나 고른다.
        pick = 0
        for i, mr in enumerate(mrows):
            if int(mr[2]) == c and 120 <= int(mr[3]) <= 180:
                pick = i
                break
        L = X.shape[2]
        H = Y.shape[1]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(range(-L + 1, 1), X[pick, ti, :], color="tab:blue", lw=1.5,
                label=f"input traffic (past {L} min)")
        ax.plot(range(1, H + 1), Y[pick], color="tab:red", lw=1.8, marker="o",
                ms=3, label=f"target traffic (next {H} min)")
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_xlabel("minutes relative to prediction time t")
        ax.set_ylabel("Traffic demand [Mbps]")
        ax.set_title(f"[{scen}] TCN sample window - cell {int(mrows[pick][2])} "
                     f"(t={int(mrows[pick][3])} min)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = outdir / "plot_sample_window.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        outs.append(p)

    return outs


def main():
    ap = argparse.ArgumentParser(description="ns-3 기반 TCN 요구트래픽 데이터셋 생성기")
    ap.add_argument("--scenarios", nargs="+", default=["일반일과"],
                    help="시나리오 이름 (Scenario/<name>.csv)")
    ap.add_argument("--days", type=int, default=1, help="시나리오당 생성 일수 (§3: 15~20)")
    ap.add_argument("--scale", type=float, default=1.0, help="인원 스케일 (검증용 <1)")
    ap.add_argument("--minutes", type=int, default=MINUTES_PER_DAY, help="하루 분 수 (기본 720)")
    ap.add_argument("--day-var", type=float, default=0.12, help="일간 인원 변동 표준편차")
    ap.add_argument("--seed", type=int, default=20250101, help="난수 시드")
    ap.add_argument("--outdir", default="tcn-dataset", help="결과 폴더")
    ap.add_argument("--keep-obs", action="store_true", help="중간 obs/waypoints CSV 보존")
    ap.add_argument("--plot", action="store_true", help="생성 후 데이터셋 확인용 PNG 그래프 생성")
    ap.add_argument("--v2-conn-delay", type=float, default=3.0, help="UE별 접속 지연 평균(분), 0~2배 균등")
    ap.add_argument("--v2-activity", type=float, default=0.7, help="접속 UE의 분당 트래픽 활성 확률")
    ap.add_argument("--v2-rate-cv", type=float, default=0.3, help="UE별 수요 배수 폭(균등 +/-)")
    ap.add_argument("--v2-pd", type=float, default=0.9, help="ISAC 검출 확률 P_D")
    args = ap.parse_args()
    v2_params = {"conn_delay": args.v2_conn_delay, "activity": args.v2_activity,
                 "rate_cv": args.v2_rate_cv, "pd": args.v2_pd}

    import numpy as np
    import random

    if not Path(rc.NS3).exists():
        sys.exit(f"error: {rc.NS3} 없음 (ns-3 루트에서 실행).")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    minutes = args.minutes
    n_slots = len(rc.SLOT_HOURS)
    keytimes = [i * (minutes / (n_slots - 1)) for i in range(n_slots)]  # 균등 분 배치

    sites_file = rc.write_sites_file(outdir)
    n_cells = len(rc.SITES)
    antenna_file = rc.generate_antenna_file(n_cells, BASELINE_ANTENNAS, outdir)
    print(f"[setup] {n_cells}셀, 고정안테나={BASELINE_ANTENNAS}, {minutes}분/일, "
          f"L={INPUT_L} H={HORIZON_H}")
    rc.build_once()

    meta = cell_meta()
    nbrs = neighbor_lists(meta)

    # 셀별 per-person 요구 rate + 저녁 배수 + 선행 시차를 파일로 써서 ns-3에 넘긴다.
    rates_file = outdir / "rates.csv"
    rrows = []
    for c in sorted(meta.keys()):
        typ = meta[c]["type"]
        eve = DORM_EVENING_MULT if typ == "기숙사" else 1.0
        # lagMin: 이 건물에 도착한 UE는 선행 시차만큼 더 기다린 뒤 접속 지연을 센다(ns-3)
        rrows.append({"cell": c, "baseRate": TYPE_RATE_MBPS.get(typ, 0.5), "eveningMult": eve,
                      "lagMin": TYPE_LAG_MIN.get(typ, 5)})
    rc.write_csv(rates_file, rrows, ["cell", "baseRate", "eveningMult", "lagMin"])

    # 셀별 시각화/디버깅용 분당 데이터셋(long CSV)도 함께 쓴다.
    long_path = outdir / "dataset_perminute.csv"
    long_cols = ["scenario", "day", "minute", "cell", "type"] + FEATURES
    long_rows = []

    records = []
    for scen in args.scenarios:
        base_rows, base_off = rc.load_scenario(scen)
        tag_s = rc.SCENARIO_TAG.get(scen, "scenario")
        for day in range(args.days):
            # hash(str)는 실행마다 바뀌므로(PYTHONHASHSEED) 고정 CRC로 시나리오를 구분한다
            drng = random.Random(args.seed + zlib.crc32(scen.encode()) % 10000 + day)
            rows, off = perturb_scenario(base_rows, drng, args.day_var) if args.days > 1 \
                else (base_rows, base_off)
            wp, n_ue = rc.generate_waypoints(
                rows, off, keytimes, outdir, scale=args.scale, seed=args.seed + day,
                filename=f"wp_{tag_s}_d{day}.csv")
            tag = f"{tag_s}_d{day}"
            rng_run = args.seed % 100000 + day + 1
            print(f"[gen] {scen} day{day} UE={n_ue} -> ns-3 dataset...")
            obs_path = run_dataset_sim(wp, sites_file, antenna_file, rates_file, minutes,
                                       rng_run, tag, outdir, v2_params)
            obs = read_obs(obs_path, n_cells, minutes + 1)
            frng = random.Random(args.seed + 7919 * (day + 1))
            rec = derive_features(obs, meta, nbrs, frng)
            for c in sorted(meta.keys()):
                records.append({"scenario": scen, "day": day, "cell": c, "rec": rec[c]})
                for m in range(len(rec[c])):
                    r = rec[c][m]
                    long_rows.append([scen, day, m, c, meta[c]["type"]] + [r[f] for f in FEATURES])
            if not args.keep_obs:
                obs_path.unlink(missing_ok=True)
                Path(wp).unlink(missing_ok=True)

    rc.write_csv(long_path, [dict(zip(long_cols, r)) for r in long_rows], long_cols)
    print(f"[done] wrote {long_path} ({len(long_rows)} rows)")

    X, Y, mrows = build_windows(records, FEATURES)
    cell_building = np.array([[c, meta[c]["name"], meta[c]["type"]]
                              for c in sorted(meta.keys())], dtype=object)
    npz_path = outdir / "tcn_dataset.npz"
    np.savez_compressed(
        npz_path,
        X=X, Y=Y,
        feature_names=np.array(FEATURES),
        target=TARGET,
        L=INPUT_L, H=HORIZON_H,
        version="v2",
        cell_building=cell_building,   # cell id -> (name, type)
        meta=np.array(mrows, dtype=object),
    )
    print(f"[done] wrote {npz_path}  X={X.shape} (N,C,L)  Y={Y.shape} (N,H)")
    print(f"       특징 {len(FEATURES)}채널: {FEATURES}")

    if args.plot:
        pngs = plot_dataset(long_path, npz_path, outdir)
        for p in pngs:
            print(f"[plot] wrote {p}")


if __name__ == "__main__":
    main()
