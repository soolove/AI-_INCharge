"""Per-cell prediction graph videos (WebM) from collect_day.py output.

  python render_cell_videos.py <scenario_dir> <repo> <out_dir> [first-last cells, default 1-18 | avg]

Writes cell_XX.webm (one cell) or, with "avg", cell_avg.webm (the 18-cell average): the day advancing in
5-min steps (one frame per agent step). At each frame only what exists at that moment is drawn: actual demand
up to now, forecasts made up to now (reaching now + 5 min), the agent's capacity up to now. Same series and
colors as render_cells.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

demo, repo, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
which = sys.argv[4] if len(sys.argv) > 4 else "1-18"
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(repo))
import build_tcn_dataset as btd  # noqa: E402

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False})

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
AGENT, FORECAST, CRIT = "#1baf7a", "#6250d6", "#d03b3b"
CMAX = btd.CELL_CMAX_MBPS

day = json.loads((demo / "meta.json").read_text())
ENV = f"   |   {day['env']}" if day.get("env") else ""   # radio setting label (e.g. e-Um 5G), if recorded
R = np.load(demo / "day_rl.npz")
meta = btd.cell_meta()
cells = sorted(meta)
names = [meta[c]["name"] for c in cells]
minutes = R["minute"]                                                   # end minute of each 5-min step
traffic = R["traffic_min"].transpose(1, 0, 2).reshape(len(cells), -1)  # (cell, minute 1..720)
tmin = np.arange(1, traffic.shape[1] + 1)
# forecast j is made at minutes[j] - 5 for t+1..t+5; ft is non-decreasing, so "made by now" is a prefix
ft = (minutes[:, None] - 5 + np.arange(1, 6)[None, :]).reshape(-1)
ft_idx = np.clip(ft.astype(int), 1, traffic.shape[1]) - 1
cap = CMAX * R["antenna"] / 16.0                                        # (step, cell), in force during the step
short = R["shortage"] > 1e-6


def hhmm(minute):
    return f"{9 + int(minute) // 60:02d}:{int(minute) % 60:02d}"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def render(c):
    """One cell's video, or the 18-cell average for c = None. The average has no shortage bands: some cell
    is short at most times, so the bands would cover most of the day."""
    avg = c is None
    act_s = traffic.mean(0) if avg else traffic[c]
    fv_all = (R["forecast"][:, :, :5].mean(1) if avg else R["forecast"][:, c, :5]).reshape(-1)
    cap_s = cap.mean(1) if avg else cap[:, c]
    fig, ax = plt.subplots(figsize=(12, 5.0), dpi=110)   # 1320 x 550: even size for yuv420p
    fig.patch.set_facecolor(SURF)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.81, bottom=0.11)
    style(ax)
    spans = [] if avg else [ax.axvspan(minutes[k] - 5, minutes[k], color=CRIT, alpha=0.12, lw=0, zorder=0,
                                       visible=False) for k in range(len(minutes))]
    actual, = ax.plot([], [], color=MUTED, lw=1.3, label="실제 요구 트래픽")
    fcast, = ax.plot([], [], color=FORECAST, lw=1.6, ls=(0, (4, 1.5)), label="TCN 예측 (1~5분 앞)")
    capl, = ax.plot([], [], color=AGENT, lw=1.8, drawstyle="steps-post", label="제공 용량 (에이전트)")
    nowl = ax.axvline(0, color=MUTED, lw=1, ls=":")
    ax.set_xlim(0, 720)
    ax.set_ylim(0, 1.12 * max(act_s.max(), fv_all.max(), cap_s.max()))
    ax.set_xticks(range(0, 721, 60), [hhmm(m) for m in range(0, 721, 60)])
    ax.set_ylabel("Mbps (셀 하나 기준)" if avg else "Mbps", color=INK2, fontsize=10)
    h, lab = ax.get_legend_handles_labels()
    if not avg:
        shade = Patch(facecolor=CRIT, alpha=0.12, label="미처리 트래픽 발생 (5분 구간)")
        h, lab = h + [shade], lab + [shade.get_label()]
    ax.legend(h, lab, loc="upper left", fontsize=9, frameon=False, ncol=4, labelcolor=INK2)
    fig.text(0.07, 0.93, f"예측 그래프: {'18개 셀 평균' if avg else names[c]}", fontsize=15, fontweight="bold",
             color=INK)
    sub = fig.text(0.07, 0.865, "", fontsize=10.5, color=INK2)

    writer = FFMpegWriter(fps=6, codec="libvpx-vp9", bitrate=1500,
                          extra_args=["-pix_fmt", "yuv420p", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4"])
    path = out / ("cell_avg.webm" if avg else f"cell_{c + 1:02d}.webm")
    with writer.saving(fig, str(path), dpi=110):
        for k, now in enumerate(minutes):
            actual.set_data(tmin[:int(now)], act_s[:int(now)])
            nf = int(np.searchsorted(ft, now + 5, side="right"))      # forecasts made at or before now
            fcast.set_data(ft[:nf], fv_all[:nf])
            capl.set_data(np.append(minutes[:k + 1] - 5, now), np.append(cap_s[:k + 1], cap_s[k]))
            if spans:
                spans[k].set_visible(bool(short[k, c]))
            nowl.set_xdata([now, now])
            npast = int(np.searchsorted(ft, now, side="right"))       # forecasts whose target minute has passed
            act = act_s[ft_idx[:npast]]
            wape = f"{100 * np.abs(fv_all[:npast] - act).sum() / act.sum():.1f}%" if act.sum() > 1e-6 else "-"
            if avg:
                sub.set_text(f"{day['scenario']}{ENV}   |   현재 시간 {hhmm(now)}   |   누적 예측 오차(WAPE) {wape}")
            else:
                dem = R["demand"][:k + 1, c].sum()
                qos = f"{100 * R['shortage'][:k + 1, c].sum() / dem:.1f}%" if dem > 1e-9 else "-"
                sub.set_text(f"{day['scenario']}{ENV}   |   현재 시간 {hhmm(now)}   |   "
                             f"누적 예측 오차(WAPE) {wape}   |   누적 QoS 부족 {qos}   |   "
                             f"현재 안테나 {int(R['antenna'][k, c])}개")
            writer.grab_frame(facecolor=SURF)
    plt.close(fig)
    return path


if which == "avg":
    targets = [None]
else:
    first, last = (int(x) for x in which.split("-"))
    targets = range(first - 1, last)
for c in targets:
    print(f"saved {render(c)}", flush=True)
