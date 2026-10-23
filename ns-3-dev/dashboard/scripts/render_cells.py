"""Per-cell full-day prediction graphs from collect_day.py output.

  python render_cells.py <scenario_dir> <repo> <out_dir>

Writes cell_01.png .. cell_18.png (one cell each), all_cells.png (18 small multiples) and cells_summary.csv.
Every series comes from the agent's run: a cell's demand depends on the control, so mixing runs would mislead.
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
from matplotlib.patches import Patch  # noqa: E402

demo, repo, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
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
R = np.load(demo / "day_rl.npz")
meta = btd.cell_meta()
cells = sorted(meta)
names = [meta[c]["name"] for c in cells]
minutes = R["minute"]                                                   # end minute of each 5-min step
traffic = R["traffic_min"].transpose(1, 0, 2).reshape(len(cells), -1)  # (cell, minute 1..720)
ft = (minutes[:, None] - 5 + np.arange(1, 6)[None, :]).reshape(-1)     # target minute of each t+1..t+5 forecast
ft_idx = np.clip(ft.astype(int), 1, traffic.shape[1]) - 1
ok = ft <= traffic.shape[1]
cap = CMAX * R["antenna"] / 16.0                                        # (step, cell), in force during the step
short = R["shortage"] > 1e-6                                            # (step, cell) unserved traffic in the step
tmin = np.arange(1, traffic.shape[1] + 1)


def hhmm(minute):
    return f"{9 + int(minute) // 60:02d}:{int(minute) % 60:02d}"


def style(ax, fs=9):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=fs)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def cell_stats(c):
    fv = R["forecast"][:, c, :5].reshape(-1)[ok]
    act = traffic[c][ft_idx[ok]]
    wape = 100 * np.abs(fv - act).sum() / act.sum() if act.sum() > 1e-6 else float("nan")
    dem, sh = R["demand"][:, c].sum(), R["shortage"][:, c].sum()
    return {"fv": fv, "wape": wape, "qos": 100 * sh / dem if dem > 1e-9 else 0.0}


def plot_cell(ax, c, st, fs=9, lw=1.0):
    for k in np.flatnonzero(short[:, c]):                               # 5-min steps with unserved traffic
        ax.axvspan(minutes[k] - 5, minutes[k], color=CRIT, alpha=0.12, lw=0, zorder=0)
    ax.plot(tmin, traffic[c], color=MUTED, lw=1.3 * lw, label="실제 요구 트래픽")
    ax.plot(ft[ok], st["fv"], color=FORECAST, lw=1.6 * lw, ls=(0, (4, 1.5)), label="TCN 예측 (1~5분 앞)")
    ax.step(minutes - 5, cap[:, c], where="post", color=AGENT, lw=1.8 * lw, label="제공 용량 (에이전트)")
    ax.set_xlim(0, 720)
    ax.set_ylim(0, 1.12 * max(traffic[c].max(), st["fv"].max(), cap[:, c].max()))
    style(ax, fs)


stats = [cell_stats(c) for c in range(len(cells))]
shade = Patch(facecolor=CRIT, alpha=0.12, label="미처리 트래픽 발생 (5분 구간)")
ENV = f"   |   {day['env']}" if day.get("env") else ""   # radio setting label (e.g. e-Um 5G), if recorded
head = f"{day['scenario']} test 날 (seed {day['day_seed']}){ENV}   |   에이전트 = PPO r10 (1:10:5) 실행"

# ---- one figure per cell
for c in range(len(cells)):
    st = stats[c]
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=110)
    fig.patch.set_facecolor(SURF)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.12)
    plot_cell(ax, c, st, fs=10)
    ax.set_xticks(range(0, 721, 60), [hhmm(m) for m in range(0, 721, 60)])
    ax.set_ylabel("Mbps", color=INK2, fontsize=10)
    wape = "-" if np.isnan(st["wape"]) else f"{st['wape']:.1f}%"
    fig.text(0.07, 0.93, f"예측 그래프: 셀 {c + 1} {names[c]}", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.07, 0.865, f"{head}   |   하루 예측 오차(WAPE) {wape}   |   이 셀의 QoS 부족 {st['qos']:.1f}%   "
             f"|   평균 안테나 {R['antenna'][:, c].mean():.1f}개", fontsize=10.5, color=INK2)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h + [shade], lab + [shade.get_label()], loc="upper left", fontsize=9, frameon=False, ncol=4,
              labelcolor=INK2)
    fig.savefig(out / f"cell_{c + 1:02d}.png", facecolor=SURF)
    plt.close(fig)

# ---- 18 small multiples (each panel has its own y-axis: cell loads differ by an order of magnitude)
fig, axes = plt.subplots(6, 3, figsize=(16, 18), dpi=100, sharex=True)
fig.patch.set_facecolor(SURF)
fig.subplots_adjust(left=0.05, right=0.985, top=0.925, bottom=0.03, hspace=0.42, wspace=0.16)
for c, ax in enumerate(axes.reshape(-1)):
    st = stats[c]
    plot_cell(ax, c, st, fs=8, lw=0.8)
    wape = "-" if np.isnan(st["wape"]) else f"{st['wape']:.1f}%"
    ax.set_title(f"셀 {c + 1} {names[c]}   WAPE {wape}   QoS 부족 {st['qos']:.1f}%", loc="left", fontsize=10,
                 color=INK, pad=4)
    ax.set_xticks(range(0, 721, 180), [hhmm(m) for m in range(0, 721, 180)])
fig.text(0.05, 0.968, "셀별 예측 그래프 (18개 셀)", fontsize=18, fontweight="bold", color=INK)
fig.text(0.05, 0.952, f"{head}   |   세로축(Mbps)은 셀마다 다름", fontsize=11, color=INK2)
h, lab = axes[0, 0].get_legend_handles_labels()
fig.legend(h + [shade], lab + [shade.get_label()], loc="upper right", bbox_to_anchor=(0.985, 0.975), ncol=4,
           fontsize=10, frameon=False, labelcolor=INK2)
fig.savefig(out / "all_cells.png", facecolor=SURF)
plt.close(fig)

# ---- summary table
with open(out / "cells_summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["셀", "이름", "하루 예측 오차 WAPE(%)", "평균 요구 트래픽(Mbps)", "최대 요구 트래픽(Mbps)",
                "평균 안테나 수", "QoS 부족(%)", "미처리 발생 5분 구간 수"])
    for c in range(len(cells)):
        st = stats[c]
        w.writerow([c + 1, names[c], "" if np.isnan(st["wape"]) else f"{st['wape']:.1f}",
                    f"{traffic[c].mean():.1f}", f"{traffic[c].max():.1f}", f"{R['antenna'][:, c].mean():.2f}",
                    f"{st['qos']:.2f}", int(short[:, c].sum())])
print(f"saved {out}: 18 cell graphs, all_cells.png, cells_summary.csv ({day['scenario']})")
