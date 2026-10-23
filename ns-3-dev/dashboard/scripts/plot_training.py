"""Final training curves of the TCN forecaster (width 64) and the PPO agent (1:10:5) from their metrics.jsonl.

  python plot_training.py <repo> <out_dir>

Writes tcn_loss.png, tcn_error.png, rl_reward.png, rl_kpi.png, rl_stability.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False})

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
TCN_C = "#6250d6"    # TCN violet, as for the forecast line in the dashboard
AGENT = "#1baf7a"    # agent green, as in the dashboard
MAXC = "#2a78d6"     # always-max blue, as in the dashboard
LIGHT = "#d9d7d2"    # raw per-update values behind a moving average

TCN_DIR = repo / "DGU_SC_Project/checkpoints/v3_5g_31d_w64_dual_proposal"
RL_DIR = repo / "rl-env_v4/runs/ppo_v2_w64_r10"
LR_SWITCH = 60       # epochs 1-60 at lr 1e-3, 61-90 resumed at lr 1e-4 (seed_0_finetune.log)
WIN = 20             # moving-average window over PPO updates


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(title, sub, ncols=1, width=12.0):
    fig, axes = plt.subplots(1, ncols, figsize=(width, 5.0), dpi=150)
    fig.patch.set_facecolor(SURF)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.74, bottom=0.12, wspace=0.22)
    fig.text(0.07, 0.93, title, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.07, 0.875, sub, fontsize=10, color=INK2)
    axes = np.atleast_1d(axes)
    for ax in axes:
        style(ax)
    return fig, axes


def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK, va="center")


def mark_best(ax, x, y, text, xytext=(-12, 22)):
    ax.scatter([x], [y], s=80, facecolor="none", edgecolor=INK, lw=1.5, zorder=6)
    ax.annotate(text, (x, y), xytext=xytext, textcoords="offset points", ha="right", fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), zorder=6)


def legend(fig, ax_or_handles, ncol):
    h, lab = ax_or_handles.get_legend_handles_labels() if hasattr(ax_or_handles, "get_legend_handles_labels") \
        else ax_or_handles
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.065, 0.845), ncol=ncol, frameon=False, fontsize=9.5,
               labelcolor=INK2)


def save(fig, name):
    fig.savefig(out / name, facecolor=SURF)
    plt.close(fig)
    print(f"saved {out / name}")


# ---------------------------------------------------------------- TCN
tcn = [json.loads(line) for line in open(TCN_DIR / "seed_0_finetune/metrics.jsonl")]
ep = np.array([r["epoch"] for r in tcn])
get = lambda split, key: np.array([r[split][key] for r in tcn])  # noqa: E731
tr_loss, va_loss = get("train", "total_loss"), get("val", "total_loss")
va_rmse, va_mae, persist = get("val", "central_rmse"), get("val", "central_mae"), get("val", "persistence_rmse")
bi = int(np.argmin(va_loss))
best_ep = int(ep[bi])
tcn_sub = (f"폭 64 TCN, 31일 데이터 (학습 943,326개 / 검증 69,876개 창), epoch 1~{LR_SWITCH} 학습률 1e-3, "
           f"{LR_SWITCH + 1}~{ep[-1]} 학습률 1e-4, 최종 모델 = 검증 손실 최저 epoch")


def lr_regions(ax, ytext):
    ax.axvline(LR_SWITCH + 0.5, color=MUTED, ls=":", lw=1)
    ax.text(LR_SWITCH / 2, ytext, "학습률 1e-3", ha="center", va="top", fontsize=9, color=INK2)
    ax.text((LR_SWITCH + ep[-1]) / 2 + 0.5, ytext, "학습률 1e-4", ha="center", va="top", fontsize=9, color=INK2)


fig, (ax,) = new_fig("TCN 학습 곡선: 손실", tcn_sub)
ax.plot(ep, tr_loss, color=MUTED, lw=1.8, label="학습")
ax.plot(ep, va_loss, color=TCN_C, lw=2.2, label="검증")
lo, hi = min(tr_loss.min(), va_loss.min()), max(tr_loss.max(), va_loss.max())
ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.12 * (hi - lo))
lr_regions(ax, hi + 0.11 * (hi - lo))
end_label(ax, ep[-1], tr_loss[-1] - 0.0004, f"학습 {tr_loss[-1]:.4f}")
end_label(ax, ep[-1], va_loss[-1] + 0.0004, f"검증 {va_loss[-1]:.4f}")
mark_best(ax, best_ep, va_loss[bi], f"최종 선택: {best_ep} epoch (검증 손실 {va_loss[bi]:.4f})", xytext=(-20, 40))
ax.set_xlim(0, ep[-1] + 9)
ax.set_xticks(range(0, ep[-1] + 1, 10))
ax.set_xlabel("epoch", color=INK2)
ax.set_ylabel("손실 (중앙값 + peak 분위 손실)", color=INK2)
legend(fig, ax, 2)
save(fig, "tcn_loss.png")

fig, (ax,) = new_fig("TCN 학습 곡선: 검증 예측 오차 (Mbps)", tcn_sub)
ax.plot(ep, va_rmse, color=TCN_C, lw=2.2, label="TCN 검증 RMSE")
ax.plot(ep, va_mae, color=TCN_C, lw=2.0, ls=(0, (5, 2)), label="TCN 검증 MAE")
ax.plot(ep, persist, color=MUTED, lw=1.6, ls=(0, (1.5, 2)), label="직전 값 유지 (persistence) RMSE")
top = max(va_rmse.max(), persist.max()) * 1.15
ax.set_ylim(0, top)
lr_regions(ax, top * 0.99)
end_label(ax, ep[-1], persist[-1], f"persistence {persist[-1]:.1f}")
end_label(ax, ep[-1], va_rmse[-1], f"RMSE {va_rmse[-1]:.1f}")
end_label(ax, ep[-1], va_mae[-1], f"MAE {va_mae[-1]:.1f}")
mark_best(ax, best_ep, va_rmse[bi], f"최종 선택 {best_ep} epoch: RMSE {va_rmse[bi]:.1f} / MAE {va_mae[bi]:.1f} Mbps",
          xytext=(-20, 34))
ax.set_xlim(0, ep[-1] + 13)
ax.set_xticks(range(0, ep[-1] + 1, 10))
ax.set_xlabel("epoch", color=INK2)
ax.set_ylabel("Mbps", color=INK2)
legend(fig, ax, 3)
save(fig, "tcn_error.png")

# ---------------------------------------------------------------- RL
rows = [json.loads(line) for line in open(RL_DIR / "metrics.jsonl")]
base = rows[0]["val_baseline_max"]                     # always-max on the same val days
tr = [r for r in rows if "train" in r]
u = np.array([r["update"] for r in tr])
vr = [r for r in rows if "val" in r]
vu = np.array([r["update"] for r in vr])
tget = lambda key: np.array([r["train"][key] for r in tr])  # noqa: E731
vget = lambda key: np.array([r["val"][key] for r in vr])    # noqa: E731
lget = lambda key: np.array([r["loss"][key] for r in tr])   # noqa: E731
vrew = vget("reward")
vb = int(np.argmax(vrew))
best_u = int(vu[vb])
rl_sub = ("PPO 에이전트 (보상 비율 1:10:5), 800 업데이트 x 하루 4개, 학습률 3e-4에서 0으로 선형 감소, "
          "검증 = 20 업데이트마다 고정 6일 (시나리오별 2일), 최종 모델 = 검증 보상 최고")


def ma(x):
    return u[WIN - 1:], np.convolve(x, np.ones(WIN) / WIN, mode="valid")


fig, (ax,) = new_fig("RL 에이전트 학습 곡선: 보상", rl_sub)
trew = tget("reward")
ax.plot(u, trew, color=LIGHT, lw=0.8, label="학습 (업데이트별)")
ax.plot(*ma(trew), color=INK2, lw=1.8, label=f"학습 ({WIN} 업데이트 이동평균)")
ax.plot(vu, vrew, color=AGENT, lw=2.0, marker="o", ms=4, label="검증")
ax.plot([0, u[-1]], [base["reward"]] * 2, color=MAXC, ls=(0, (5, 2)), lw=1.5, label="상시 최대 (검증)")
ax.set_ylim(None, max(vrew.max(), trew.max()) + 0.06)
end_label(ax, u[-1], base["reward"], f"상시 최대 {base['reward']:.3f}")
end_label(ax, vu[-1], vrew[-1], f"검증 {vrew[-1]:.3f}")
mark_best(ax, best_u, vrew[vb], f"최종 선택: {best_u} 업데이트 (검증 보상 {vrew[vb]:.4f})", xytext=(-30, 36))
ax.set_xlim(0, u[-1] + 95)
ax.set_xlabel("업데이트 (1 업데이트 = 하루 4개 학습)", color=INK2)
ax.set_ylabel("보상 (높을수록 좋음)", color=INK2)
legend(fig, ax, 4)
save(fig, "rl_reward.png")

fig, axes = new_fig("RL 에이전트 학습 중 전력 절감률과 QoS 부족", rl_sub, ncols=2, width=14.0)
for ax, key, title, best_xy in ((axes[0], "saving_pct", "전력 절감률 (상시 최대 대비, %)", (-30, 22)),
                                (axes[1], "shortage_pct", "QoS 부족 (미처리 트래픽 비율, %)", (-50, -24))):
    mx, my = ma(tget(key))
    ax.plot(mx, my, color=INK2, lw=1.6, label=f"학습 ({WIN} 업데이트 이동평균)")
    v = vget(key)
    ax.set_ylim(0, 1.15 * max(my.max(), v.max()))
    ax.plot(vu, v, color=AGENT, lw=2.0, marker="o", ms=4, label="검증")
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=6)
    ax.set_xlim(0, u[-1] + 120)
    ax.set_xlabel("업데이트", color=INK2)
    end_label(ax, vu[-1], v[-1], f"검증 {v[-1]:.1f}%")
    mark_best(ax, best_u, v[vb], f"최종 선택 {best_u}: {v[vb]:.1f}%", xytext=best_xy)
axes[0].plot([0, u[-1]], [30, 30], color=INK2, ls=(0, (1.5, 2)), lw=1.2, label="KPI2 목표 30%")
end_label(axes[0], u[-1], 30, "KPI2 목표 30%")
axes[1].plot([0, u[-1]], [base["shortage_pct"]] * 2, color=MAXC, ls=(0, (5, 2)), lw=1.5, label="상시 최대 (검증)")
end_label(axes[1], u[-1], base["shortage_pct"], f"상시 최대 {base['shortage_pct']:.2f}%")
h0, l0 = axes[0].get_legend_handles_labels()
h1, l1 = axes[1].get_legend_handles_labels()
legend(fig, (h0 + h1[-1:], l0 + l1[-1:]), 4)
save(fig, "rl_kpi.png")

fig, axes = new_fig("RL 에이전트 학습 안정성 지표", rl_sub, ncols=4, width=16.0)
for ax, key, title, log in ((axes[0], "entropy", "정책 엔트로피 (entropy)", False),
                            (axes[1], "kl", "근사 KL (approx. KL)", False),
                            (axes[2], "clip_frac", "클리핑 비율 (clip fraction)", False),
                            (axes[3], "value_loss", "가치 손실 (value loss, 로그)", True)):
    x = lget(key)
    ax.plot(u, x, color=LIGHT, lw=0.8)
    ax.plot(*ma(x), color=AGENT, lw=1.8)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=6)
    ax.set_xlabel("업데이트", color=INK2)
    if log:
        ax.set_yscale("log")
legend(fig, ([Line2D([], [], color=LIGHT, lw=1.5), Line2D([], [], color=AGENT, lw=2)],
             ["업데이트별", f"{WIN} 업데이트 이동평균"]), 2)
save(fig, "rl_stability.png")
print(f"TCN best epoch {best_ep} (val loss {va_loss[bi]:.4f}); RL best update {best_u} (val reward {vrew[vb]:.4f})")
