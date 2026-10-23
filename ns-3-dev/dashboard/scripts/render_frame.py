"""Render dashboard preview frames from collect_day.py output.

  python render_frame.py <demo_dir> <repo> <out.png|out.gif|out.mp4> [HH:MM | HH:MM-HH:MM]

Left: campus map (the agent's antennas). Right: cumulative energy and QoS shortage (top row) and the
current 5-min step's energy and QoS shortage (bottom row) for always-max, reactive and the agent.
The 18-cell average prediction graph is a separate video (render_cell_videos.py ... avg).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

demo, repo, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
when = sys.argv[4] if len(sys.argv) > 4 else "11:30"
sys.path.insert(0, str(repo))
import build_tcn_dataset as btd  # noqa: E402

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False})

# reference palette (light): surface, text, categorical slots 1-3, sequential blue steps
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
COL = {"max": "#2a78d6", "reactive": "#eb6834", "rl": "#1baf7a"}
NAME = {"max": "상시 최대", "reactive": "반응형", "rl": "에이전트"}
ANT_FILL = {2: "#cde2fb", 4: "#9ec5f4", 8: "#5598e7", 16: "#256abf"}
CRIT = "#d03b3b"
RADIUS = {2: 40.0, 4: 55.0, 8: 70.0, 16: 85.0}

D = {m: np.load(demo / f"day_{m}.npz") for m in ("max", "reactive", "rl")}
day = json.loads((demo / "meta.json").read_text())
ENV = f"   |   {day['env']}" if day.get("env") else ""   # radio setting label (e.g. e-Um 5G), if recorded
meta = btd.cell_meta()
cells = sorted(meta)
pos = np.array([[meta[c]["x"], meta[c]["y"]] for c in cells])
minutes = D["rl"]["minute"]                      # end minute of each 5-min step (5, 10, ..., 720)
E_DAY = D["max"]["energy"].sum()                 # always-max energy over the whole day = 100%
P_MAX = D["max"]["energy"].mean()                # always-max energy in one step = 100%


def step_shortage(m):
    """Unserved share of the campus demand in each 5-min step (%)."""
    return 100 * D[m]["shortage"].sum(1) / np.maximum(D[m]["demand"].sum(1), 1e-9)


# the current-step QoS axis is fixed for the whole day (up to the evening peak) so it does not jump
QNOW_TOP = max(1.0, 1.1 * max(float(step_shortage(m).max()) for m in D))

# UE waypoints -> position at any minute (linear between waypoints, like WaypointMobilityModel)
wp = {}
with open(demo / "day_rl.wp.csv") as fh:
    for row in csv.DictReader(fh):
        wp.setdefault(int(row["ueId"]), []).append((float(row["t"]), float(row["x"]), float(row["y"]), int(row["on"])))
for k in wp:
    wp[k].sort()


def ue_positions(minute):
    xs, ys = [], []
    for pts in wp.values():
        t = [p[0] for p in pts]
        i = np.searchsorted(t, minute, side="right") - 1
        if i < 0 or pts[i][3] == 0:
            continue
        if i + 1 < len(pts):
            t0, x0, y0, _ = pts[i]
            t1, x1, y1, _ = pts[i + 1]
            a = (minute - t0) / (t1 - t0) if t1 > t0 else 0.0
            xs.append(x0 + a * (x1 - x0))
            ys.append(y0 + a * (y1 - y0))
        else:
            xs.append(pts[i][1])
            ys.append(pts[i][2])
    return np.array(xs), np.array(ys)


def hhmm(minute):
    return f"{9 + int(minute) // 60:02d}:{int(minute) % 60:02d}"


bg = plt.imread(demo.parent / "campus_bg.png")


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def end_labels(ax, ends, top, x):
    """Direct labels at the line ends; labels closer than 7% of the axis height are pushed apart."""
    ends.sort()
    gap = 0.07 * top
    ends[0][0] = max(ends[0][0], 0.035 * top)   # keep a 0% label off the x-axis line
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    over = ends[-1][0] - 0.98 * top
    if over > 0:
        for row in ends:
            row[0] -= over
    for y, label in ends:
        ax.annotate(label, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=INK, va="center")


def draw(fig, step):
    fig.clear()
    fig.patch.set_facecolor(SURF)
    now = minutes[step]
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 1], left=0.03, right=0.985,
                          top=0.88, bottom=0.07, wspace=0.12, hspace=0.42)
    fig.text(0.03, 0.955, "안테나 제어 rApp 대시보드", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.03, 0.915, f"{day['scenario']}{ENV}   |   {hhmm(now)}", fontsize=11, color=INK2)

    # ---- (a) campus map: the agent's antennas, coverage radius, UEs
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(bg, extent=[0, 600, 0, 550], origin="upper", alpha=0.55, zorder=0)
    ant = D["rl"]["antenna"][step].astype(int)
    short = D["rl"]["shortage"][step]
    for c in range(len(cells)):
        a = ant[c]
        ax.add_patch(Circle(pos[c], RADIUS[a], facecolor=ANT_FILL[a], edgecolor="#184f95", lw=0.8, alpha=0.2, zorder=1))
    ux, uy = ue_positions(now)
    ax.scatter(ux, uy, s=3, color=INK2, alpha=0.55, lw=0, zorder=2)
    for c in range(len(cells)):
        a = ant[c]
        hot = short[c] > 1e-6
        ax.scatter(*pos[c], s=46, marker="^", color=CRIT if hot else "#104281", zorder=3, lw=1.2, edgecolor=SURF)
        ax.annotate(f"{a}", pos[c], xytext=(5, 4), textcoords="offset points", fontsize=8.5,
                    fontweight="bold", color=CRIT if hot else INK, zorder=4)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 550)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title("캠퍼스 맵: 셀별 활성 안테나 수와 커버리지", loc="left", fontsize=13, color=INK, pad=8)

    # ---- right: cumulative performance (top row) and the current 5-min step (bottom row), three methods.
    # Two small charts per row, one measure each (no dual axis).
    k = step + 1
    t = minutes[:k]
    cum_e, cum_q, now_e, now_q = {}, {}, {}, {}
    for m in ("max", "reactive", "rl"):
        en, sh, dm = D[m]["energy"][:k], D[m]["shortage"][:k].sum(1), D[m]["demand"][:k].sum(1)
        cum_e[m] = 100 * np.cumsum(en) / E_DAY
        cum_q[m] = 100 * np.cumsum(sh) / np.maximum(np.cumsum(dm), 1e-9)
        now_e[m] = 100 * en / P_MAX
        now_q[m] = step_shortage(m)[:k]
    top_row = gs[0, 1].subgridspec(1, 2, wspace=0.28)
    bottom_row = gs[1, 1].subgridspec(1, 2, wspace=0.28)
    panels = (  # (grid cell, title, series, y top, label decimals, line width, start from 0 at 09:00)
        (top_row[0, 0], "누적 전력량 (상시 최대 하루 = 100%)", cum_e, 110.0, 0, 2.0, True),
        (top_row[0, 1], "누적 QoS 부족 (미처리 트래픽 비율)", cum_q,
         max(1.0, 1.1 * max(float(v.max()) for v in cum_q.values())), 1, 2.0, False),
        (bottom_row[0, 0], "현재 전력량 (5분 단위, 상시 최대 = 100%)", now_e, 110.0, 0, 1.5, False),
        (bottom_row[0, 1], "현재 QoS 부족 (5분 단위 미처리 비율)", now_q, QNOW_TOP, 1, 1.5, False),
    )
    for spec, title, series, top, dec, lw, from_zero in panels:
        ax = fig.add_subplot(spec)
        style(ax)
        ends = []
        for m, v in series.items():
            ax.plot(np.r_[0, t] if from_zero else t, np.r_[0, v] if from_zero else v, color=COL[m], lw=lw)
            ends.append([float(v[-1]), f"{NAME[m]} {v[-1]:.{dec}f}%"])
        end_labels(ax, ends, top, t[-1])
        ax.set_xlim(0, 720 + 230)
        ax.set_xticks(range(0, 721, 240), [hhmm(x) for x in range(0, 721, 240)])
        ax.set_ylim(0, top)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=6)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))  # exact at 1 decimal
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=dec))
    # one legend for the four charts; the summary gives both methods' absolute values, because the ratio of
    # two small cumulative shortages early in the day swings by hundreds of percent
    fig.legend([Line2D([], [], color=COL[m], lw=2.5) for m in COL], [NAME[m] for m in COL], loc="upper left",
               bbox_to_anchor=(0.523, 0.99), ncol=3, frameon=False, fontsize=10.5, labelcolor=INK2)
    save = {m: 100 * (1 - D[m]["energy"][:k].sum() / D["max"]["energy"][:k].sum()) for m in ("reactive", "rl")}
    fig.text(0.53, 0.925, f"누적 성능   전력 절감: 에이전트 {save['rl']:.1f}% (반응형 {save['reactive']:.1f}%)   |   "
             f"QoS 부족: 에이전트 {cum_q['rl'][-1]:.1f}% (반응형 {cum_q['reactive'][-1]:.1f}%)",
             fontsize=12, color=INK, fontweight="bold")
    save_now = {m: 100 * (1 - D[m]["energy"][k - 1] / D["max"]["energy"][k - 1]) for m in ("reactive", "rl")}
    fig.text(0.53, 0.47, f"현재 성능   전력 절감: 에이전트 {save_now['rl']:.1f}% (반응형 {save_now['reactive']:.1f}%)   |   "
             f"QoS 부족: 에이전트 {now_q['rl'][-1]:.1f}% (반응형 {now_q['reactive'][-1]:.1f}%)",
             fontsize=12, color=INK, fontweight="bold")


if "-" in when:
    a, b = when.split("-")
    to_min = lambda s: (int(s[:2]) - 9) * 60 + int(s[3:])  # noqa: E731
    steps = [i for i, m in enumerate(minutes) if to_min(a) <= m <= to_min(b)]
else:
    target = (int(when[:2]) - 9) * 60 + int(when[3:])
    steps = [int(np.argmin(np.abs(minutes - target)))]

fig = plt.figure(figsize=(16, 9), dpi=110)
if out.suffix == ".png":
    draw(fig, steps[0])
    fig.savefig(out, facecolor=SURF)
else:
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    anim = FuncAnimation(fig, lambda i: draw(fig, steps[i]), frames=len(steps))
    writer = PillowWriter(fps=4) if out.suffix == ".gif" else FFMpegWriter(fps=6, bitrate=4000)
    anim.save(out, writer=writer, savefig_kwargs={"facecolor": SURF})
print(f"saved {out} ({len(steps)} frame(s))")
