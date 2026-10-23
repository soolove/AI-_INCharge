"""KPI1: TCN forecast MAPE on the test days (proposal target <= 10%).

Definition used for the reported KPI1 (first computed 2026-10-12, now kept as a script):
- test days of the checkpoint's split (train_dual.py config; days 13-14 for the provided model,
  day 2 for the QUICK=1 run of run_tcn_v3.sh), central (p50) head, all 15 horizons pooled;
- MAPE over the target values Y >= 0.25 x C_max (the capacity of 4 active antennas),
  because MAPE is unstable near zero traffic;
- also reported: t+1 / t+15 MAPE, per-scenario MAPE, WAPE over all values, and the
  persistence baseline (last observed traffic) MAPE over the same values Y >= 0.25 x C_max.

  python eval_kpi1.py --npz ../tcn-dataset-v3-5g-31d/tcn_dataset.npz \
      --ckpt checkpoints/v3_5g_31d_w64_dual_proposal/seed_0_finetune/best.pt --cmax 580
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from tcn_dual import DualHeadTCN, width_from_arch

TEST_DAYS = (13, 14)  # only for checkpoints without split_days in their config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cmax", type=float, default=580.0, help="cell capacity C_max in dataset units (Mbps)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg, s = ck["config"], ck["normalization"]
    model = DualHeadTCN(cfg["input_channels"],
                        channels=width_from_arch(cfg.get("architecture"))).to(args.device).eval()
    model.load_state_dict(ck["model"])
    stats = {k: v.to(args.device) for k, v in s.items()}

    test_days = tuple(cfg.get("split_days", {}).get("test", TEST_DAYS))
    d = np.load(args.npz, allow_pickle=True)
    meta = d["meta"]
    idx = np.flatnonzero(np.isin(meta[:, 1].astype(int), test_days))
    x = d["X"][idx][:, cfg["feature_indices"], :].astype(np.float32)
    y = d["Y"][idx].astype(np.float64)
    traffic_ch = cfg["feature_names"].index("Traffic_demand_mbps")

    preds = []
    with torch.inference_mode():
        for a in range(0, len(x), 8192):
            xb = (torch.from_numpy(x[a:a + 8192]).to(args.device) - stats["x_mean"]) / stats["x_std"]
            central = model(xb)[0] * stats["y_std"] + stats["y_mean"]
            preds.append(central.cpu().numpy())
    p = np.concatenate(preds).astype(np.float64)

    err = np.abs(p - y)
    hi = y >= 0.25 * args.cmax
    persist = np.abs(x[:, traffic_ch, -1:].astype(np.float64) - y)
    print(f"checkpoint: {args.ckpt} (epoch {ck['epoch']})")
    print(f"dataset:    {args.npz}, test days {test_days}, {len(idx)} windows")
    print(f"KPI1 MAPE (Y >= {0.25 * args.cmax:g} Mbps = 0.25 x C_max): {100 * np.mean(err[hi] / y[hi]):.1f}%"
          f"  (share of values {100 * hi.mean():.1f}%, of traffic {100 * y[hi].sum() / y.sum():.1f}%)")
    print(f"  t+1 {100 * np.mean(err[:, 0][hi[:, 0]] / y[:, 0][hi[:, 0]]):.1f}%, "
          f"t+15 {100 * np.mean(err[:, 14][hi[:, 14]] / y[:, 14][hi[:, 14]]):.1f}%")
    scen = meta[idx, 0]
    for sc in dict.fromkeys(scen.tolist()):
        k = hi & (scen == sc)[:, None]
        print(f"  {sc}: {100 * np.mean(err[k] / y[k]):.1f}%")
    print(f"WAPE (all values): {100 * err.sum() / y.sum():.1f}%")
    print(f"persistence MAPE (Y >= {0.25 * args.cmax:g} Mbps): {100 * np.mean(persist[hi] / y[hi]):.1f}%")


if __name__ == "__main__":
    main()
