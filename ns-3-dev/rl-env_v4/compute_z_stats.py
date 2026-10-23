"""RL State에 넣는 TCN 맥락 임베딩 z의 표준화 통계(z_stats.json)를 만든다.

z는 TCN 마지막 시점의 ReLU 출력(차원 = TCN 폭)이라 가끔 크게 튄다. 그래서 환경은 z를
TCN 학습 구간에서 구한 차원별 평균·표준편차로 표준화하고 +-clip에서 자른다
(ns3_antenna_env.py). TCN을 새로 학습하면 z도 바뀌므로 이 파일을 다시 만들어야 한다.

학습 구간 날짜와 입력 채널은 체크포인트 config(split_days, feature_indices)에서 읽는다.

  python rl-env_v4/compute_z_stats.py                        (기본 TCN -> rl-env_v4/z_stats.json)
  python rl-env_v4/compute_z_stats.py --ckpt <best.pt> --out <json>
  # 최소 실행: QUICK=1 run_tcn_v3.sh의 TCN -> run_exp_eum5g.sh quick이 읽는 위치
  python rl-env_v4/compute_z_stats.py --ckpt DGU_SC_Project/checkpoints/quick/seed_0_finetune/best.pt \\
      --npz tcn-dataset-quick/tcn_dataset.npz --out DGU_SC_Project/checkpoints/quick/z_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tcn_forecaster import DEFAULT_CKPT, TcnForecaster  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def day_spec(days):
    """[0, 1, 2, 5] -> '0-2,5'"""
    parts, start = [], None
    for i, d in enumerate(days):
        if start is None:
            start = d
        if i + 1 == len(days) or days[i + 1] != d + 1:
            parts.append(f"{start}-{d}" if d > start else f"{d}")
            start = None
    return ",".join(parts)


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="train_dual.py로 학습한 TCN best.pt")
    ap.add_argument("--npz", default="tcn-dataset-v3-5g-31d/tcn_dataset.npz",
                    help="그 TCN을 학습한 데이터셋 (build_tcn_dataset.py 출력)")
    ap.add_argument("--out", default="rl-env_v4/z_stats.json")
    ap.add_argument("--clip", type=float, default=5.0, help="표준화한 z를 자르는 범위 (+-clip)")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, npz = REPO / args.ckpt, REPO / args.npz
    fc = TcnForecaster(ckpt, device=args.device)
    cfg = torch.load(ckpt, map_location="cpu", weights_only=True)["config"]
    train_days = sorted(cfg["split_days"]["train"])

    with np.load(npz, allow_pickle=True) as d:
        idx = np.flatnonzero(np.isin(d["meta"][:, 1].astype(int), train_days))
        x = d["X"][idx][:, cfg["feature_indices"], :].astype(np.float32)
    print(f"TCN {rel(ckpt)} (z {fc.z_dim}차원), 학습 구간 {day_spec(train_days)}일 창 {len(x):,}개")

    s = fc.stats
    zs = []
    with torch.inference_mode():
        for a in range(0, len(x), args.batch):
            xb = torch.from_numpy(x[a:a + args.batch]).to(fc.device)
            zs.append(fc.model.tcn((xb - s["x_mean"]) / s["x_std"])[:, :, -1].cpu().numpy())
    z = np.concatenate(zs)

    # float32 그대로 계산한다. 제공된 z_stats.json(에이전트가 학습한 값)이 이 방식이라 GPU에서는
    # 비트 단위로 같다 (float64로 구하면 표준편차가 최대 0.8% 다르다).
    mean = z.mean(axis=0)
    std = np.maximum(z.std(axis=0), 1e-6)
    clipped = float(np.mean(np.abs((z - mean) / std) > args.clip))
    stats = {"mean": mean.tolist(), "std": std.tolist(),
             "clip": args.clip,
             "source": f"{rel(npz)} TCN train days {day_spec(train_days)}, {rel(ckpt)}",
             "n_windows": len(z), "z_dim": int(z.shape[1]), "clipped_frac": clipped}
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"clip 밖 비율 {100 * clipped:.2f}%, 저장: {out}")


if __name__ == "__main__":
    main()
