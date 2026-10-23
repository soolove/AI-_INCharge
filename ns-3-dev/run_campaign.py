#!/usr/bin/env python3
"""
run_campaign.py - 캠퍼스 시나리오 공용 도우미 (build_tcn_dataset.py와 rl-env_v4가 import)

- 캠퍼스 사이트 표(SITES)와 ns-3용 sites.csv(건물/야외 구분 포함)
- 시나리오 인원표(Scenario/<name>.csv) 읽기와 UE별 웨이포인트(ueId,t,x,y,on) 생성
- 캠퍼스 배경 지도 자르기(캠퍼스 밖 좌표 샘플링에 씀)
- ns-3 프로그램(cttc-nr-demo) 빌드

파일 이름은 예전 시스템레벨 캠페인 드라이버 때 이름이다(이 이름으로 import 한다).
"""

import csv
import math
import random
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
NS3 = str(REPO_ROOT / "ns3")
NS3_PROGRAM = "cttc-nr-demo"

# 캠퍼스 배경 이미지(경계+건물). Map_info/건물 좌표.jpg 의 축 안쪽(플롯) 영역을
# 잘라(CROP) 좌표 범위(EXTENT)에 맞춰 배경으로 사용한다.
# 배경: 범례가 제거된 캠퍼스 이미지. 플롯 영역(CROP)과 그것이 대응하는 실제 좌표
# (EXTENT)는 이미지 축 프레임/눈금에서 검출해 사이트 좌표와 겹치도록 보정한 값.
MAP_IMG = REPO_ROOT.parent / "Map_info" / "건물 좌표(legend제거).jpeg"
# 축 여백을 빼고 정확히 데이터 [0,600] x [0,550] 영역만 자른다 (음수좌표 방지).
# x=0->px95, x=600->px1006, y=550->row21, y=0->row864.
MAP_CROP = (95, 21, 1006, 864)            # 데이터 영역 픽셀 (left, upper, right, lower)
MAP_EXTENT = [0.0, 600.0, 0.0, 550.0]     # 대응 좌표 (xmin, xmax, ymin, ymax)
TYPE_EN = {"강의동": "Lecture", "도서관": "Library",
           "기숙사": "Dorm", "이벤트": "Event"}

# ---------------------------------------------------------------------------
# 시나리오(움직이는 UE) 설정: Scenario/<name>.csv 의 건물별/시간대별 인원표를
# 읽어 각 UE가 시간대에 따라 건물 사이(및 캠퍼스 밖)를 이동하도록 웨이포인트를
# 만든다. 시간대는 09:00~21:00, 90분 간격 9개 슬롯.
# ---------------------------------------------------------------------------
SCENARIO_DIR = REPO_ROOT.parent / "Scenario"
SLOT_COLS = ["t0900", "t1030", "t1200", "t1330", "t1500", "t1630", "t1800", "t1930", "t2100"]
SLOT_HOURS = [9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0]
# 시나리오 이름(한글) -> 출력 파일용 ASCII 태그
SCENARIO_TAG = {"일반일과": "normal", "시험기간": "exam", "축제": "festival"}
BUILDING_SCATTER_R = 18.0    # 건물 주변 UE 산포 반경 (m)


# 실제 캠퍼스(동국대 서울) 사이트: (name, type, gnb, x, y)
#   gnb = 설치된 실제 기지국 하드웨어 수 (고정). 활성 안테나 수와는 별개(제어값).
#   각 사이트 = 1개 셀 = 1개 제어단위(건물).
SITES = [
    ("정보문화관P", "강의동", 6, 100.7, 417.9),
    ("정보문화관Q", "강의동", 1, 70.4, 435.6),
    ("원흥관", "강의동", 2, 107.9, 345.4),
    ("신공학관", "강의동", 12, 81.4, 253.1),
    ("과학관", "강의동", 4, 196.4, 150.6),
    ("만해관", "강의동", 2, 300.7, 216.5),
    ("혜화관", "강의동", 4, 374.1, 188.4),
    ("경영관", "강의동", 2, 457.8, 145.9),
    ("본관", "강의동", 2, 175.0, 299.2),
    ("중앙도서관", "도서관", 7, 127.7, 219.8),
    ("학술관", "도서관", 3, 538.1, 246.0),
    ("학술문화관", "도서관", 1, 520.1, 243.8),
    ("남산학사", "기숙사", 4, 36.3, 271.1),
    ("만해광장", "이벤트", 2, 234.2, 256.5),
    ("문화관", "이벤트", 4, 502.2, 241.6),
    ("대운동장", "이벤트", 3, 275.0, 90.0),
    ("체육관", "이벤트", 2, 239.2, 432.2),
    ("학생회관", "이벤트", 2, 134.2, 470.1),
]

# 건물이 아닌 야외 사이트(광장, 운동장). 이 사이트 근처 UE에는 ns-3 건물 침투 손실(--o2iLossDb)을
# 적용하지 않는다. 나머지 사이트는 건물로, 산포 반경(BUILDING_SCATTER_R) 안의 UE를 실내로 본다.
OUTDOOR_SITES = {"만해광장", "대운동장"}


def write_sites_file(outdir):
    """캠퍼스 사이트 CSV(name,type,gnb,x,y,indoor)를 쓰고 경로를 돌려준다.
    indoor = 1(건물) / 0(야외 사이트, OUTDOOR_SITES)."""
    rows = [{"name": n, "type": t, "gnb": g, "x": x, "y": y,
             "indoor": 0 if n in OUTDOOR_SITES else 1} for (n, t, g, x, y) in SITES]
    path = outdir / "sites.csv"
    write_csv(path, rows, ["name", "type", "gnb", "x", "y", "indoor"])
    return path


def crop_campus_bg(outdir):
    """건물 좌표.jpg를 플롯 영역으로 잘라 campus_bg.png로 저장하고 경로를 돌려준다.
    없으면 None."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if not MAP_IMG.exists():
        return None
    Image.open(MAP_IMG).convert("RGB").crop(MAP_CROP).save(outdir / "campus_bg.png")
    return outdir / "campus_bg.png"


def build_once():
    print(f"[build] {NS3} build {NS3_PROGRAM}")
    subprocess.run([NS3, "build", NS3_PROGRAM], cwd=REPO_ROOT, check=True)


def generate_antenna_file(n_cells, n_antennas, outdir):
    """모든 셀의 활성 안테나 수를 n_antennas로 둔 CSV(cell,numAntennas)를 쓰고 경로를 돌려준다.
    데이터셋 모드의 고정 안테나 기준선이다(C++ CoverageModel)."""
    rows = [{"cell": c, "numAntennas": n_antennas} for c in range(1, n_cells + 1)]
    path = outdir / "antenna.csv"
    write_csv(path, rows, ["cell", "numAntennas"])
    return path


# ===========================================================================
# 시나리오(움직이는 UE) 파이프라인
# ===========================================================================
def load_scenario(name):
    """Scenario/<name>.csv 를 읽어 (건물행 리스트, 캠퍼스밖 인원 리스트)를 돌려준다.
    건물행 = (건물명, 유형, [슬롯별 인원 9개]). 건물명은 SITES 이름과 1:1 매칭."""
    path = SCENARIO_DIR / f"{name}.csv"
    if not path.exists():
        sys.exit(f"error: 시나리오 파일 없음: {path}")
    rows = []
    off = [0] * len(SLOT_COLS)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            counts = [int(r[c]) for c in SLOT_COLS]
            if r["building"] == "__offcampus__":
                off = counts
            else:
                rows.append((r["building"], r["type"], counts))
    return rows, off


def make_offcampus_sampler(outdir, seed):
    """캠퍼스 경계(파란 실선) '밖'의 좌표를 뽑아주는 샘플러를 돌려준다.
    campus_bg.png에서 초록/채도 픽셀 -> 구멍 채우기로 캠퍼스 내부 마스크를 만들고,
    그 여집합(맵 범위 내)에서 균일 샘플링한다. 반환값은 실좌표 (x, y)."""
    bg = crop_campus_bg(outdir)
    x0, x1, y0, y1 = MAP_EXTENT
    rng = random.Random(seed + 1000)
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        bg = None
    if bg is None:
        # 마스크를 못 만들면 맵 가장자리 띠에서 샘플링 (근사).
        def sample_edge():
            side = rng.random()
            m = 40.0
            if side < 0.5:
                return (rng.uniform(x0, x1), rng.uniform(y0, y0 + m))
            return (rng.uniform(x0, x0 + m), rng.uniform(y0, y1))
        return sample_edge

    arr = np.asarray(Image.open(bg).convert("RGB"), dtype=np.int16)
    Rc, Gc, Bc = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mx = np.maximum(np.maximum(Rc, Gc), Bc)
    mn = np.minimum(np.minimum(Rc, Gc), Bc)
    sat = mx - mn
    campus = ((Gc > Rc + 4) & (Gc > Bc + 4)) | (sat > 25)
    try:
        from scipy.ndimage import binary_closing, binary_fill_holes
        campus = binary_closing(campus, iterations=3)
        campus = binary_fill_holes(campus)
    except ImportError:
        pass
    H, W = campus.shape

    def sample():
        for _ in range(2000):
            x = rng.uniform(x0, x1)
            y = rng.uniform(y0, y1)
            col = int(round((x - x0) / (x1 - x0) * (W - 1)))
            row = int(round((y1 - y) / (y1 - y0) * (H - 1)))
            if not campus[row, col]:
                return (x, y)
        return (x, y)  # 실패 시 마지막 표본

    return sample


def generate_waypoints(scenario_rows, off_counts, keytimes, outdir,
                       scale=1.0, seed=7, filename="waypoints.csv"):
    """시나리오 인원표를 각 UE의 웨이포인트(ueId,t,x,y,on)로 변환해 CSV로 쓴다.
    - 각 슬롯에서 건물별 인원만큼 UE를 그 건물 근처(고정 산포)에 배치.
    - 남는(=캠퍼스밖) UE는 경계 밖 고정 지점에 배치.
    - keytimes[i] = 슬롯 i 의 시각(분). 좌표는 실좌표(C++의 실좌표 gNB와 정합)."""
    site_xy = {n: (x, y) for (n, t, g, x, y) in SITES}
    building_order = [s[0] for s in SITES]
    n_slots = len(SLOT_HOURS)

    # 슬롯별 인원(스케일 적용). 건물 순서는 SITES 고정.
    scaled = []
    for t in range(n_slots):
        d = {name: int(round(cnts[t] * scale)) for (name, typ, cnts) in scenario_rows}
        d["__offcampus__"] = int(round(off_counts[t] * scale))
        scaled.append(d)
    n_ue = max(sum(d.values()) for d in scaled)
    if n_ue == 0:
        sys.exit("error: 시나리오 인원이 0 입니다 (scale 확인).")

    # UE별 고정 산포 오프셋(건물 근처)과 고정 캠퍼스밖 지점.
    sampler = make_offcampus_sampler(outdir, seed)
    rng = random.Random(seed)
    offs = []
    homes = []
    for _ in range(n_ue):
        rr = BUILDING_SCATTER_R * math.sqrt(rng.random())
        th = rng.random() * 2.0 * math.pi
        offs.append((rr * math.cos(th), rr * math.sin(th)))
        homes.append(sampler())

    # 슬롯 t 를 위한 웨이포인트를 시각 st 에 emit. on=1 이면 건물(캠퍼스 안, 셀
    # 집계 대상), on=0 이면 캠퍼스밖(집계 제외). prefix 방식으로 UE를
    # 건물->...->캠퍼스밖 순서로 채운다.
    def emit(slot, st, out):
        idx = 0
        for name in building_order:
            c = scaled[slot][name]
            bx, by = site_xy[name]
            for _ in range(c):
                if idx >= n_ue:
                    break
                ox, oy = offs[idx]
                out.append({"ueId": idx, "t": st,
                            "x": round(bx + ox, 2), "y": round(by + oy, 2), "on": 1})
                idx += 1
        while idx < n_ue:  # 남은 UE = 캠퍼스밖 (고정 지점)
            hx, hy = homes[idx]
            out.append({"ueId": idx, "t": st, "x": round(hx, 2), "y": round(hy, 2), "on": 0})
            idx += 1

    rows = []
    for t in range(n_slots):
        emit(t, keytimes[t], rows)

    path = outdir / filename
    write_csv(path, rows, ["ueId", "t", "x", "y", "on"])
    return path, n_ue


def write_csv(path, rows, fieldnames):
    # lineterminator="\n": Unix newlines so the C++ CSV parsers don't see a
    # trailing '\r' on the last column.
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
