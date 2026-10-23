"""반응형 제어 비교군의 임계값을 val 날에서 고른다.

반응형 제어는 현재 PRB 사용률이 up보다 높으면 안테나를 한 단계 올리고, down보다 낮으면
한 단계 내린다(제안서 4절 비교군 2). 임계값을 임의로 정하면 일부러 약한 상대를 세운 것이
될 수 있으므로, RL이 best.pt를 고를 때와 같은 val 날과 같은 보상으로 격자 탐색해 가장
좋은 조합을 쓴다. test 날은 쓰지 않는다.

  python rl-env_v4/tune_reactive.py --out rl-env_v4/runs/reactive_tuned.json
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campus_env import CampusRLEnv, split_days  # noqa: E402
from ns3_antenna_env import RewardConfig  # noqa: E402
from stub_policy import threshold  # noqa: E402
from tcn_forecaster import TcnForecaster  # noqa: E402
from train_ppo import evaluate_days, fmt, mean_summary  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="rl-env_v4/runs/reactive_tuned.json")
    ap.add_argument("--ups", default="0.005,0.0075,0.01,0.015,0.02,0.03,0.05,0.075,0.1,0.15,0.2")
    ap.add_argument("--downs", default="0.001,0.0025,0.005,0.0075,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--val-days", type=int, default=2)
    ap.add_argument("--minutes", type=int, default=720)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--sinr-threshold", type=float, default=-5.0)
    ap.add_argument("--static-power", type=float, default=0.3)
    # 보상 비율 1:10:5를 합이 1이 되게 나눈 값 (에이전트 학습과 같은 가중치)
    ap.add_argument("--w-energy", type=float, default=0.0625)
    ap.add_argument("--w-qos", type=float, default=0.625)
    ap.add_argument("--w-overload", type=float, default=0.3125)
    ap.add_argument("--u-th", type=float, default=0.8)
    ap.add_argument("--workdir", default="rl-env_v4/run/tune",
                    help="에피소드 입력 파일 작업 폴더. 조정 여러 개를 동시에 돌리려면 서로 다르게 준다")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = RewardConfig(w_energy=args.w_energy, w_qos=args.w_qos, w_overload=args.w_overload,
                       u_th=args.u_th, static_power_ratio=args.static_power)
    env = CampusRLEnv(TcnForecaster(device=args.device), workdir=args.workdir,
                      minutes=args.minutes, scale=args.scale, sinr_threshold=args.sinr_threshold,
                      reward_cfg=cfg, include_z=False)
    days = split_days("val", args.val_days)
    grid = [(u, d) for u, d in itertools.product(
        [float(x) for x in args.ups.split(",")], [float(x) for x in args.downs.split(",")]) if d < u]
    results = []
    try:
        for i, (up, down) in enumerate(grid, 1):
            m = mean_summary(evaluate_days(env, lambda o, u=up, d=down: (threshold(o, up=u, down=d), None),
                                           days))
            results.append({"up": up, "down": down, "val": m})
            print(f"[{i:2d}/{len(grid)}] up={up:g} down={down:g}  {fmt(m)}", flush=True)
    finally:
        env.close()

    best = max(results, key=lambda r: r["val"]["reward"])
    print(f"\n선택: up={best['up']}, down={best['down']}  {fmt(best['val'])}")
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"up": best["up"], "down": best["down"], "val": best["val"],
                               "reward_config": vars(cfg), "grid": results},
                              ensure_ascii=False, indent=2))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
