"""Train the dual-head TCN (central p50 series + peak p90) on the campus NPZ.

Design fixed in the project spec:
  - central head -> per-minute traffic, pinball loss at tau_central (default 0.5)
  - peak head    -> the actual 15-minute maximum m = max_h(Y_h),
                    pinball loss at tau_peak (default 0.9)
  - total loss   = L_central + peak_weight * L_peak     (both on standardized targets)

Existing single-head checkpoints are never touched: use a NEW --output folder.
The code reads channel count and horizon from the data, so a regenerated dataset
of the same shape can be trained by re-running with the new --npz, no edits.

Metrics (validation, Mbps, raw output; no clipping / margin):
  central : mae, rmse, per-horizon rmse
  peak    : coverage P(m <= p_hat), deficit max(m - p_hat, 0), over max(p_hat - m, 0)
  persistence_rmse : "future = current value" baseline, for reference
best.pt is selected by validation total_loss by default; every per-epoch metric is
logged to metrics.jsonl so a constrained selection (spec sec.9) can be done post-hoc.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from tcn_dual import DualHeadTCN, pinball_loss, width_from_arch

# Default split of tcn-dataset-v3-5g-31d (the same as run_tcn_v3.sh).
SPLIT_DAYS = {"train": [*range(11), *range(15, 31)], "val": [11, 12], "test": [13, 14]}
ARCH = "tcn-dual-{width}x4-k3-d1-2-4-8-dropout0.1-L60-H15-central15-peak1"
NORM = "train-window-channel-zscore-central-peak-v1"


def parse_days(spec):
    """'0-10,15-30' or '11,12' -> sorted list of day indices."""
    days = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        days.update(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    return sorted(days)


def load_data(path, split_days=None):
    """Load the NPZ and the window indices of each split.

    split_days None uses SPLIT_DAYS. Every listed day must be present in the dataset;
    days in no split are left out.
    """
    with np.load(path, allow_pickle=True) as data:
        x, y, meta = data["X"], data["Y"], data["meta"]
        names = data["feature_names"].tolist()
        h = int(data["H"])
        if int(data["L"]) != 60:
            raise ValueError("Expected L=60")
        if str(data["target"]) != "Traffic_demand_mbps":
            raise ValueError("Expected Traffic_demand_mbps target")
    if x.shape != (len(y), len(names), 60) or y.shape != (len(x), h):
        raise ValueError(f"Invalid shapes: X={x.shape}, Y={y.shape}")
    if meta.shape != (len(x), 4) or x.dtype != np.float32 or y.dtype != np.float32:
        raise ValueError("Expected float32 X/Y and [N,4] metadata")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Dataset contains NaN or infinity")
    days = meta[:, 1].astype(int)
    if split_days is None:
        split_days = SPLIT_DAYS
    listed = [d for value in split_days.values() for d in value]
    if len(listed) != len(set(listed)):
        raise ValueError(f"Splits overlap: {split_days}")
    missing = sorted(set(listed) - set(days.tolist()))
    if missing:
        raise ValueError(f"Days {missing} are not in the dataset")
    indices = {key: np.flatnonzero(np.isin(days, value))
               for key, value in split_days.items()}
    scenarios = set(meta[:, 0])
    for key, idx in indices.items():
        if not len(idx) or set(meta[idx, 0]) != scenarios:
            raise ValueError(f"Empty split or missing scenario: {key}")
    return x, y, meta, names, indices, h


def normalization(x, y, train_idx, feature_indices):
    """Train-only statistics. X in chunks to avoid copying the full array."""
    sums = np.zeros(x.shape[1], dtype=np.float64)
    squares = np.zeros_like(sums)
    for start in range(0, len(train_idx), 4096):
        chunk = x[train_idx[start:start + 4096]].astype(np.float64)
        sums += chunk.sum(axis=(0, 2))
        squares += np.square(chunk).sum(axis=(0, 2))
    count = len(train_idx) * x.shape[2]
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0)).clip(1e-6)
    ytr = y[train_idx].astype(np.float64)
    mtr = ytr.max(axis=1)  # 15-minute maxima on the train split
    stats = {
        "x_mean": torch.tensor(mean, dtype=torch.float32)[None, :, None],
        "x_std": torch.tensor(std, dtype=torch.float32)[None, :, None],
        "y_mean": torch.tensor(ytr.mean(), dtype=torch.float32),
        "y_std": torch.tensor(max(ytr.std(), 1e-6), dtype=torch.float32),
        "m_mean": torch.tensor(mtr.mean(), dtype=torch.float32),
        "m_std": torch.tensor(max(mtr.std(), 1e-6), dtype=torch.float32),
    }
    stats["x_mean"] = stats["x_mean"][:, feature_indices, :]
    stats["x_std"] = stats["x_std"][:, feature_indices, :]
    return stats


def run_epoch(model, loader, stats, device, optimizer=None, max_batches=0,
              feature_indices=None, tau_central=0.5, tau_peak=0.9, peak_weight=1.0):
    model.train(optimizer is not None)
    total_loss_sum = 0.0
    sq = ab = base = 0.0
    horizon_sq = torch.zeros(15, device=device)
    covered = deficit = over = 0.0
    values = 0  # count of Y elements
    samples = 0  # count of windows
    with torch.set_grad_enabled(optimizer is not None):
        for batch, (x, y) in enumerate(loader):
            if max_batches and batch >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            inputs = x if feature_indices is None else x[:, feature_indices, :]
            central, peak = model((inputs - stats["x_mean"]) / stats["x_std"])
            m = y.max(dim=1).values
            cy = (y - stats["y_mean"]) / stats["y_std"]
            cm = (m - stats["m_mean"]) / stats["m_std"]
            loss = (pinball_loss(central, cy, tau_central)
                    + peak_weight * pinball_loss(peak, cm, tau_peak))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; previous checkpoint is preserved")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            # De-standardize to Mbps for reporting.
            central_mbps = central.detach() * stats["y_std"] + stats["y_mean"]
            peak_mbps = peak.detach() * stats["m_std"] + stats["m_mean"]
            err = central_mbps - y
            sq += err.square().sum().item()
            ab += err.abs().sum().item()
            horizon_sq += err.square().sum(dim=0)
            base += (x[:, 0, -1, None] - y).square().sum().item()
            peak_err = peak_mbps - m
            covered += (peak_mbps >= m).sum().item()
            deficit += (-peak_err).clamp_min(0).sum().item()
            over += peak_err.clamp_min(0).sum().item()
            total_loss_sum += loss.item() * len(y)
            values += y.numel()
            samples += len(y)
    if not samples:
        raise ValueError("No samples in loader")
    return {"total_loss": total_loss_sum / samples,
            "central_rmse": (sq / values) ** 0.5,
            "central_mae": ab / values,
            "central_horizon_rmse": (horizon_sq / samples).sqrt().cpu().tolist(),
            "peak_coverage": covered / samples,
            "peak_deficit_mbps": deficit / samples,
            "peak_over_mbps": over / samples,
            "persistence_rmse": (base / values) ** 0.5}


@torch.inference_mode()
def evaluate(model, x, y, meta, idx, stats, device, feature_indices,
             cell_building, batch_size, high_load_threshold):
    """Full validation/test report with percentiles, per-building and high-load."""
    model.eval()
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(Subset(ds, idx), batch_size=batch_size)
    c_all, p_all = [], []
    for x_b, _ in loader:
        x_b = x_b.to(device)
        inp = x_b if feature_indices is None else x_b[:, feature_indices, :]
        central, peak = model((inp - stats["x_mean"]) / stats["x_std"])
        c_all.append((central * stats["y_std"] + stats["y_mean"]).cpu().numpy())
        p_all.append((peak * stats["m_std"] + stats["m_mean"]).cpu().numpy())
    c_hat = np.concatenate(c_all)          # [n,15] Mbps
    p_hat = np.concatenate(p_all)          # [n]    Mbps
    yv = y[idx]                            # [n,15]
    m = yv.max(axis=1)                     # [n]
    cur = x[idx][:, 0, -1]                 # current traffic for persistence
    cells = meta[idx, 2].astype(int)
    cb = {int(r[0]): str(r[1]) for r in cell_building} if len(cell_building) else {}

    def block(sel):
        c, p, yy, mm, cc = c_hat[sel], p_hat[sel], yv[sel], m[sel], cur[sel]
        if len(mm) == 0:
            return None
        err = c - yy
        defi = np.clip(mm - p, 0, None)
        out = {
            "windows": int(len(mm)),
            "central_rmse": float(np.sqrt((err ** 2).mean())),
            "central_mae": float(np.abs(err).mean()),
            "peak_coverage": float((p >= mm).mean()),
            "peak_deficit_mean": float(defi.mean()),
            "peak_deficit_p90_p95_max": [float(np.percentile(defi, 90)),
                                          float(np.percentile(defi, 95)),
                                          float(defi.max())],
            "peak_over_mean": float(np.clip(p - mm, 0, None).mean()),
            "persistence_rmse": float(np.sqrt(((cc[:, None] - yy) ** 2).mean())),
        }
        return out

    report = {
        "overall": block(np.ones(len(m), dtype=bool)),
        "central_horizon_rmse": [float(np.sqrt(((c_hat[:, h] - yv[:, h]) ** 2).mean()))
                                  for h in range(yv.shape[1])],
        "high_load": block(m > high_load_threshold),
        "high_load_threshold": high_load_threshold,
        "building": {},
    }
    for c in sorted(set(cells.tolist())):
        name = cb.get(c, f"cell_{c}")
        report["building"][name] = block(cells == c)
    return report


def save_checkpoint(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Run folder for training (not needed with --evaluate)")
    parser.add_argument("--epochs", type=int, default=60, help="Total epochs incl. resumed")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated feature subset (e.g. proposal set). "
                             "Traffic_demand_mbps is required. Omit to use all channels.")
    parser.add_argument("--central-quantile", type=float, default=0.5)
    parser.add_argument("--peak-quantile", type=float, default=0.9)
    parser.add_argument("--peak-weight", type=float, default=1.0, help="lambda on the peak loss")
    parser.add_argument("--high-load-threshold", type=float, default=464.0,
                        help="Mbps threshold for the high-load report block (analysis only; "
                             "0.8 x C_max 580 Mbps)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate", type=Path, help="Evaluate a checkpoint without training")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--max-batches", type=int, default=0, help="Smoke check only")
    parser.add_argument("--train-days", type=str, default=None,
                        help="Custom split such as '0-10,15-30', given together with --val-days "
                             "and --test-days. Default: days 0-10,15-30 / 11-12 / 13-14 "
                             "(SPLIT_DAYS). --evaluate reuses the checkpoint's split.")
    parser.add_argument("--val-days", type=str, default=None)
    parser.add_argument("--test-days", type=str, default=None)
    parser.add_argument("--channels", type=int, default=64,
                        help="Width of the 4 temporal blocks (and of z); 64 = the TCN used in this "
                             "project (run_tcn_v3.sh). Stored in config['architecture'].")
    args = parser.parse_args()
    if not args.evaluate and args.output is None:
        parser.error("--output is required for training")
    if (args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0
            or args.max_batches < 0 or not (0 < args.central_quantile < 1)
            or not (0 < args.peak_quantile < 1) or args.peak_weight < 0):
        parser.error("Invalid hyperparameters")
    if args.resume and args.evaluate:
        parser.error("Choose --resume or --evaluate")
    if not args.evaluate and not args.resume and any(
            (args.output / n).exists() for n in ("last.pt", "best.pt", "metrics.jsonl")):
        parser.error("Checkpoints already exist: use --resume or a different --output")
    if args.resume and args.resume.resolve() != (args.output / "last.pt").resolve():
        parser.error("Resume from the last.pt in --output")
    custom = [args.train_days, args.val_days, args.test_days]
    if any(custom) and not all(custom):
        parser.error("Give --train-days, --val-days and --test-days together")
    split_days = None
    if all(custom):
        split_days = {"train": parse_days(args.train_days), "val": parse_days(args.val_days),
                      "test": parse_days(args.test_days)}
    elif args.evaluate:
        saved_split = torch.load(args.evaluate, map_location="cpu",
                                 weights_only=True).get("config", {}).get("split_days")
        if saved_split and saved_split != SPLIT_DAYS:
            split_days = {k: list(v) for k, v in saved_split.items()}

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    x, y, meta, names, indices, h = load_data(args.npz, split_days)
    if h != 15:
        raise ValueError("This head layout assumes H=15")
    if names[0] != "Traffic_demand_mbps":
        raise ValueError("Persistence baseline requires traffic as channel 0")
    if "Isac_density" not in names or len(set(names)) != len(names):
        raise ValueError("Expected unique feature names including Isac_density")
    with args.npz.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        dataset_hash = digest.hexdigest()
    with np.load(args.npz, allow_pickle=True) as data:
        version = str(data["version"]) if "version" in data else "legacy"
        cell_building = data["cell_building"].tolist() if "cell_building" in data else []

    if args.features:
        requested = [f.strip() for f in args.features.split(",") if f.strip()]
        unknown = [f for f in requested if f not in names]
        if unknown:
            parser.error(f"Unknown feature(s): {unknown}; available: {names}")
        if "Traffic_demand_mbps" not in requested:
            parser.error("Traffic_demand_mbps is required in --features")
        feature_indices = sorted(names.index(f) for f in requested)
    else:
        feature_indices = list(range(len(names)))
    selected = [names[i] for i in feature_indices]
    config = {"input_channels": len(selected), "feature_names": selected,
              "source_feature_names": names, "feature_indices": feature_indices,
              "dataset_version": version,
              "architecture": ARCH.format(width=args.channels),
              "normalization": NORM,
              "dataset_sha256": dataset_hash,
              "split_days": split_days if split_days is not None else SPLIT_DAYS,
              "batch_size": args.batch_size, "lr": args.lr,
              "loss": "dual_pinball", "tau_central": args.central_quantile,
              "tau_peak": args.peak_quantile, "peak_weight": args.peak_weight,
              "seed": args.seed, "max_batches": args.max_batches}

    if args.evaluate:
        ckpt = torch.load(args.evaluate, map_location="cpu", weights_only=True)
        saved = ckpt.get("config", {})
        if saved.get("loss") != "dual_pinball":
            raise ValueError("Use a checkpoint from this dual-head trainer")
        if saved["dataset_sha256"] != dataset_hash or saved["source_feature_names"] != names:
            raise ValueError("Evaluation dataset differs from training dataset")
        model = DualHeadTCN(saved["input_channels"],
                            channels=width_from_arch(saved.get("architecture"))).to(device)
        model.load_state_dict(ckpt["model"])
        stats = {k: v.to(device) for k, v in ckpt["normalization"].items()}
        report = evaluate(model, x, y, meta, indices[args.split], stats, device,
                          saved["feature_indices"], cell_building, args.batch_size,
                          args.high_load_threshold)
        print(json.dumps({"split": args.split, "epoch": ckpt["epoch"],
                          "report": report}, ensure_ascii=False), flush=True)
        return

    model = DualHeadTCN(len(selected), channels=args.channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, best = 0, float("inf")
    sampler_rng = torch.Generator().manual_seed(args.seed)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        saved_config = dict(ckpt["config"])
        saved_config["lr"] = args.lr
        if saved_config != config:
            raise ValueError("Dataset, features or training settings differ from checkpoint")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        print(f"Resume learning rate: {ckpt['config']['lr']} -> {args.lr}", flush=True)
        stats = ckpt["normalization"]
        start, best = ckpt["epoch"], ckpt["best_val_total_loss"]
        torch.set_rng_state(ckpt["rng_state"])
        sampler_rng.set_state(ckpt["sampler_rng_state"])
        if device.type == "cuda" and ckpt["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    else:
        stats = normalization(x, y, indices["train"], feature_indices)
    stats = {k: v.to(device) for k, v in stats.items()}
    if args.epochs <= start:
        parser.error(f"Checkpoint already completed {start} epochs; increase --epochs")

    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loaders = {k: DataLoader(Subset(dataset, idx), batch_size=args.batch_size,
                             shuffle=(k == "train"),
                             generator=sampler_rng if k == "train"
                             else torch.Generator().manual_seed(0))
               for k, idx in indices.items() if k != "test"}
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Device={device}; dataset={version}; features={selected}; RF={model.receptive_field}; "
          f"tau_c={args.central_quantile}; tau_p={args.peak_quantile}; lambda={args.peak_weight}; "
          f"sizes={ {k: len(i) for k, i in indices.items()} }", flush=True)
    if args.max_batches:
        print("SMOKE CHECK: limited batches, not full validation.", flush=True)

    kw = dict(tau_central=args.central_quantile, tau_peak=args.peak_quantile,
              peak_weight=args.peak_weight)
    for epoch in range(start + 1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], stats, device, optimizer,
                          args.max_batches, feature_indices, **kw)
        val = run_epoch(model, loaders["val"], stats, device, max_batches=args.max_batches,
                        feature_indices=feature_indices, **kw)
        improved = val["total_loss"] < best
        best = min(best, val["total_loss"])
        state = {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "config": config,
                 "normalization": {k: v.cpu() for k, v in stats.items()},
                 "best_val_total_loss": best, "train_metrics": train, "val_metrics": val,
                 "rng_state": torch.get_rng_state(),
                 "sampler_rng_state": sampler_rng.get_state(),
                 "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        if improved:
            save_checkpoint(args.output / "best.pt", state)
        save_checkpoint(args.output / "last.pt", state)
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "train": train, "val": val}) + "\n")
        print(f"Epoch {epoch}: val loss={val['total_loss']:.4f}; "
              f"central RMSE={val['central_rmse']:.4f}, MAE={val['central_mae']:.4f}; "
              f"peak cover={val['peak_coverage']:.3f}, deficit={val['peak_deficit_mbps']:.4f}, "
              f"over={val['peak_over_mbps']:.4f} Mbps; persist RMSE={val['persistence_rmse']:.4f}; "
              f"saved last.pt", flush=True)


if __name__ == "__main__":
    main()
