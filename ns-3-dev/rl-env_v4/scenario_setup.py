"""RL 환경 실행에 필요한 시나리오 입력 파일을 만든다.

ns-3 쪽 rlEnv 모드는 dataset 모드와 같은 입력(sites/waypoints/rates)을 받으므로,
build_tcn_dataset.py가 쓰는 것과 동일한 헬퍼로 생성해 관측 분포를 일치시킨다.
"""

import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import build_tcn_dataset as btd  # noqa: E402
import run_campaign as rc  # noqa: E402


def prepare(outdir, scenario="일반일과", minutes=720, scale=1.0, seed=7, day=0,
            day_var=0.0, wp_name=None):
    """시나리오 입력 파일 3종을 outdir에 만들고 절대 경로를 담은 dict를 돌려준다.

    day_var > 0 이면 build_tcn_dataset.py와 같은 방식으로 건물별·시간대별 인원에
    일간 변동을 준다(학습 데이터셋 기본값 0.12). 변동 난수는 (seed, day)로 정해지므로
    같은 값이면 같은 날이 다시 만들어진다.

    @return dict(sites, waypoints, rates, n_ue, n_cells)
    """
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    sites = Path(rc.write_sites_file(outdir)).resolve()
    meta = btd.cell_meta()

    rates = outdir / "rates.csv"
    rows = []
    for c in sorted(meta.keys()):
        typ = meta[c]["type"]
        eve = btd.DORM_EVENING_MULT if typ == "기숙사" else 1.0
        # lagMin: 건물 유형별 선행 시차(도착 -> 첫 트래픽, 분). ns-3가 접속 지연에 더한다
        rows.append({"cell": c, "baseRate": btd.TYPE_RATE_MBPS.get(typ, 0.5), "eveningMult": eve,
                     "lagMin": btd.TYPE_LAG_MIN.get(typ, 5)})
    rc.write_csv(rates, rows, ["cell", "baseRate", "eveningMult", "lagMin"])

    n_slots = len(rc.SLOT_HOURS)
    keytimes = [i * (minutes / (n_slots - 1)) for i in range(n_slots)]
    base_rows, base_off = rc.load_scenario(scenario)
    if day_var > 0:
        base_rows, base_off = btd.perturb_scenario(
            base_rows, random.Random(seed * 100003 + day), day_var)
    tag = rc.SCENARIO_TAG.get(scenario, "scenario")
    wp, n_ue = rc.generate_waypoints(
        base_rows, base_off, keytimes, outdir, scale=scale, seed=seed + day,
        filename=wp_name or f"wp_rl_{tag}_d{day}.csv")

    return {
        "sites": str(sites),
        "waypoints": str(Path(wp).resolve()),
        "rates": str(rates),
        "n_ue": n_ue,
        "n_cells": len(meta),
    }
