"""PPO로 셀별 안테나 제어 정책을 학습한다 (State = TCN 예측, 맥락 임베딩 z, 현재 상태).

  1 에피소드 = 시뮬레이션 하루(144 step x 5분). 매 에피소드마다 train 시드 범위에서
  (시나리오, 날)을 무작위로 골라 ns-3를 새로 띄운다.
  1 갱신 = 에피소드 --episodes-per-update개(576 step)를 모은 뒤 PPO 갱신.
  검증 = --val-every 갱신마다 고정 val 날들(시나리오 3개 x --val-days)에서 결정적 정책으로
         평가하고, 하루 보상(셀 평균)이 가장 높은 정책을 best.pt로 저장.

설계
  - 셀별 보상과 셀별 가치로 GAE를 셀마다 계산한다. 18개 셀의 보상 합은 캠퍼스 전체 보상과
    같아서 목표는 그대로이고, 각 셀의 행동에 자기 몫의 이득이 돌아간다.
  - PPO 확률 비율을 셀마다 따로 계산한다 (18개 셀 확률의 곱으로 계산하면 clip에 자주 걸린다).
  - 엔트로피 보너스는 셀당 평균 엔트로피에 건다 (18개 셀의 합에 걸면 18배 강해진다).
  - 학습률을 마지막 갱신까지 선형으로 0까지 줄인다.
  - 보상에 과부하 페널티(엔진스펙 8절, 사용률 0.8 초과)가 들어간다.
  - 공간 할인: 셀 c의 학습 신호 = r_c + alpha * (최근접 3셀 보상의 합). 셀별 보상만 쓰면
    셀 c가 안테나를 줄여 이웃에 떠넘긴 부족분·과부하를 c가 보지 못한다. 다수 신호등 제어
    RL(Chu et al., IEEE T-ITS 2020)의 spatial discount와 같은 방식이다. alpha=0이면 순수 셀별.
  - 기록하는 보상은 섞기 전의 셀 평균(= 캠퍼스 전체 보상 / 18)이다.

TCN은 고정이며 정책만 학습한다. 기본은 제공된 TCN과 rl-env_v4/z_stats.json이고, 다른 TCN은
--tcn-ckpt와 그 TCN의 z 통계 --z-stats(compute_z_stats.py)로 준다. 두 경로는 체크포인트의
args에 남아 evaluate.py가 같은 TCN으로 평가한다. tensorboard가 있으면 <out>/tb에 곡선을
기록하고, tqdm이 있으면 버퍼가 차는 과정을 막대로 보여준다(stderr).

  bash rl-env_v4/run_exp_eum5g.sh train     # weights 1:10:5 as decimals plus --reward-scale 16
  bash rl-env_v4/run_exp_eum5g.sh quick     # 최소 실행: 2회 갱신(갱신당 1일), val 시나리오당 1일
  tensorboard --logdir rl-env_v4/runs
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campus_env import CampusRLEnv, split_days  # noqa: E402
from ns3_antenna_env import RewardConfig  # noqa: E402
from policy import SharedCellActorCritic  # noqa: E402
from tcn_forecaster import DEFAULT_CKPT, TcnForecaster  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SUMMARY_KEYS = ("reward", "saving_pct", "shortage_pct", "drop_pct", "overload_pct",
                "term_energy", "term_qos", "term_overload",
                "drop_sampled_pct", "outage_events", "outage_events_long", "outage_sample_sec")


def run_episode(env, policy_fn, options=None, collect=False, on_step=None):
    """하루를 끝까지 돌린다. policy_fn(obs) -> (행동 (18,), 기록용 dict 또는 None).

    반환: (요약 dict, 학습용 궤적 dict 또는 None). 요약의 보상과 세 항은 셀 평균의 하루 합이며
    reward = -(term_energy + term_qos + term_overload) 이다.
    """
    obs, info = env.reset(options=options)
    n = obs["raw"].shape[0]
    s = {"scenario": info["scenario"], "day_seed": info["day_seed"], "steps": 0, "minutes": 0,
         "reward": 0.0, "energy": 0.0, "shortage": 0.0, "demand": 0.0, "ue": 0.0, "outage": 0.0,
         "overload": 0.0, "term_energy": 0.0, "term_qos": 0.0, "term_overload": 0.0,
         "connected_samples": 0.0, "outage_samples": 0.0, "outage_events": 0.0,
         "outage_events_long": 0.0, "outage_sample_sec": float("nan")}
    traj = {"state": [], "action": [], "logp": [], "value": [], "reward": []} if collect else None
    while True:
        action, rec = policy_fn(obs)
        if collect:
            for k in ("state", "action", "logp", "value"):
                traj[k].append(rec[k])
        obs, reward, terminated, truncated, inf = env.step(action)
        m = inf["minutes"]
        s["steps"] += 1
        s["minutes"] += m
        s["reward"] += reward
        s["energy"] += inf["energy"] * m
        s["overload"] += inf["overload_frac"] * m
        s["shortage"] += float(inf["shortage_mbps"].sum())
        s["demand"] += float(inf["demand_mbps"].sum())
        s["ue"] += inf["ue_total"]
        s["outage"] += inf["outage_ue_total"]
        for k in ("term_energy", "term_qos", "term_overload"):
            s[k] += inf[k] / n
        for k in ("connected_samples", "outage_samples", "outage_events", "outage_events_long"):
            s[k] += inf[k]
        s["outage_sample_sec"] = inf["outage_sample_sec"]
        if collect:
            traj["reward"].append(inf["cell_reward"])
        if on_step is not None:
            on_step()
        if terminated or truncated:
            break
    return summarize(s), traj


def summarize(s):
    mins = max(s["minutes"], 1)
    return {"scenario": s["scenario"], "day_seed": s["day_seed"], "steps": s["steps"],
            "reward": s["reward"],
            "saving_pct": 100.0 * (1.0 - s["energy"] / mins),
            "shortage_pct": 100.0 * s["shortage"] / s["demand"] if s["demand"] > 0 else 0.0,
            "drop_pct": 100.0 * s["outage"] / s["ue"] if s["ue"] > 0 else 0.0,
            "overload_pct": 100.0 * s["overload"] / mins,
            "term_energy": s["term_energy"], "term_qos": s["term_qos"],
            "term_overload": s["term_overload"],
            # 분 사이 샘플 기준 끊김(간격 outage_sample_sec초): 끊김 시간 비율, 하루 끊김 횟수,
            # 그중 기준 시간 이상 이어진 횟수(간격이 기준 이상이면 nan). 샘플이 분당 1개면
            # drop_sampled_pct는 drop_pct와 같다.
            "drop_sampled_pct": (100.0 * s["outage_samples"] / s["connected_samples"]
                                 if s["connected_samples"] > 0 else 0.0),
            "outage_events": s["outage_events"], "outage_events_long": s["outage_events_long"],
            "outage_sample_sec": s["outage_sample_sec"]}


def mean_summary(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in SUMMARY_KEYS}


def make_policy_fn(model, device, deterministic):
    @torch.no_grad()
    def fn(obs):
        state = torch.as_tensor(obs["state"], device=device).unsqueeze(0)
        action, logp, value = model.act(state, deterministic=deterministic)
        a = action.squeeze(0).cpu().numpy()
        return a, {"state": obs["state"], "action": a,
                   "logp": logp.squeeze(0).cpu().numpy(), "value": value.squeeze(0).cpu().numpy()}
    return fn


def gae(rewards, values, gamma, lam):
    """셀별 GAE. rewards, values: (T, N). 마지막 step 뒤는 하루가 끝나 리셋되므로 가치 0."""
    rewards = np.asarray(rewards, np.float32)
    values = np.asarray(values, np.float32)
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1], np.float32)
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def neighbor_matrix(env, n):
    """M[c, j] = 1 if j가 c의 최근접 3셀. 환경의 이웃 정의(build_tcn_dataset)를 그대로 쓴다."""
    m = np.zeros((n, n), np.float32)
    for c, idx in enumerate(env.env._nbrs):
        m[c, idx] = 1.0
    return m


def spatial_mix(rewards, nbr, alpha):
    """rewards (T, N) -> r_c + alpha * sum_{j in N(c)} r_j."""
    return rewards + alpha * rewards @ nbr.T


def ppo_update(model, optimizer, batch, args, device):
    """셀별 확률 비율·이득으로 PPO clip 손실을 계산한다. 모든 텐서는 (T, N)."""
    state = torch.as_tensor(np.stack(batch["state"]), device=device)
    action = torch.as_tensor(np.stack(batch["action"]), device=device, dtype=torch.long)
    old_logp = torch.as_tensor(np.stack(batch["logp"]), device=device)
    adv = torch.as_tensor(np.concatenate(batch["adv"]), device=device)
    ret = torch.as_tensor(np.concatenate(batch["ret"]), device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(adv)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "kl": 0.0}
    count = 0
    for _ in range(args.epochs):
        for idx in torch.randperm(n, device=device).split(args.minibatch):
            logp, entropy, value = model.evaluate(state[idx], action[idx])
            ratio = torch.exp(logp - old_logp[idx])
            a = adv[idx]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
            vl = 0.5 * (value - ret[idx]).pow(2).mean()
            ent = entropy.mean()
            loss = pg + args.vf_coef * vl - args.ent_coef * ent
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                stats["policy_loss"] += pg.item()
                stats["value_loss"] += vl.item()
                stats["entropy"] += ent.item()
                stats["clip_frac"] += ((ratio - 1).abs() > args.clip).float().mean().item()
                stats["kl"] += (old_logp[idx] - logp).mean().item()
            count += 1
    return {k: v / max(count, 1) for k, v in stats.items()}


class Progress:
    """tqdm 진행 막대. 설치돼 있지 않거나 끈 경우에는 아무것도 하지 않는다.

    막대는 stderr로만 그린다. 학습 로그(stdout)만 파일에 남기려면
    `python rl-env_v4/train_ppo.py ... | tee 로그` 처럼 2>&1 없이 실행한다.
    """

    def __init__(self, enabled):
        self.tqdm = None
        if enabled:
            try:
                from tqdm import tqdm
                self.tqdm = tqdm
            except ImportError:
                print("[경고] tqdm이 없어 진행 막대를 건너뜁니다 (pip install tqdm)", flush=True)

    def bar(self, total, desc, unit):
        if self.tqdm is None:
            return None
        return self.tqdm(total=total, desc=desc, unit=unit, leave=False, file=sys.stderr,
                         mininterval=0.5, dynamic_ncols=True)


def make_writer(logdir, enabled):
    """tensorboard SummaryWriter. 설치돼 있지 않으면 경고만 하고 기록 없이 학습한다."""
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[경고] tensorboard가 없어 곡선 기록을 건너뜁니다 "
              "(pip install tensorboard 'protobuf==3.20.3')", flush=True)
        return None
    return SummaryWriter(log_dir=str(logdir))


def log_scalars(writer, prefix, values, step):
    if writer is None:
        return
    for k, v in values.items():
        if v == v:  # nan(예: 분당 1회 샘플의 긴 끊김 횟수처럼 측정 불가)은 기록하지 않는다
            writer.add_scalar(f"{prefix}/{k}", v, step)


def evaluate_days(env, policy_fn, days, progress=None, desc="val"):
    bar = progress.bar(len(days), desc, "일") if progress else None
    rows = []
    for sc, sd in days:
        rows.append(run_episode(env, policy_fn, {"scenario": sc, "day_seed": sd})[0])
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return rows


def fmt(m):
    return (f"보상 {m['reward']:7.3f} | 절감 {m['saving_pct']:5.1f}% | 부족 {m['shortage_pct']:5.2f}% "
            f"| 끊김 {m['drop_pct']:5.2f}% | 과부하 {m['overload_pct']:5.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="rl-env_v4/runs/ppo")
    ap.add_argument("--include-z", action=argparse.BooleanOptionalAction, default=True,
                    help="TCN 맥락 임베딩 z(차원 = TCN 폭, 지금 64, z_stats.json으로 표준화)를 State에 넣기 "
                         "(끄려면 --no-include-z)")
    ap.add_argument("--tcn-ckpt", default=None,
                    help="State 예측에 쓰는 TCN best.pt (ns-3-dev 기준 경로). 없으면 제공된 TCN")
    ap.add_argument("--z-stats", default=None,
                    help="그 TCN의 z 표준화 통계 (compute_z_stats.py 출력). 없으면 rl-env_v4/z_stats.json")
    ap.add_argument("--updates", type=int, default=800)
    ap.add_argument("--episodes-per-update", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-lr-anneal", action="store_true", help="학습률을 줄이지 않고 고정")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01, help="셀당 평균 엔트로피에 거는 계수")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--spatial-alpha", type=float, default=1.0,
                    help="학습 신호에 섞는 이웃 3셀 보상의 비율 (0 = 순수 셀별 보상)")
    ap.add_argument("--reward-scale", type=float, default=16.0,
                    help="가치 함수 학습용으로 셀 보상에 곱하는 상수 (기록되는 보상은 원래 값). "
                         "기본 16 = 정수 비율 1:10:5의 합")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--val-days", type=int, default=2, help="val 날 수 (시나리오당)")
    ap.add_argument("--minutes", type=int, default=720)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--sinr-threshold", type=float, default=-5.0)
    ap.add_argument("--static-power", type=float, default=0.3)
    # 보상 비율 1:10:5를 합이 1이 되게 나눈 값. 기록되는 보상이 0~1 범위가 된다
    ap.add_argument("--w-energy", type=float, default=0.0625)
    ap.add_argument("--w-qos", type=float, default=0.625)
    ap.add_argument("--w-overload", type=float, default=0.3125)
    ap.add_argument("--u-th", type=float, default=0.8, help="인접 셀 과부하 기준 사용률 (엔진스펙 8절)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-tensorboard", action="store_true", help="tensorboard 기록 끄기")
    ap.add_argument("--no-progress", action="store_true", help="tqdm 진행 막대 끄기")
    ap.add_argument("--verbose-ns3", action="store_true",
                    help="ns3-ai가 에피소드마다 찍는 실행 메시지를 그대로 보여주기 (디버깅용)")
    args = ap.parse_args()

    out = (REPO / args.out).resolve()
    if (out / "metrics.jsonl").exists():
        ap.error(f"{out}에 이미 학습 기록이 있습니다. 다른 --out을 쓰세요")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    reward_cfg = RewardConfig(w_energy=args.w_energy, w_qos=args.w_qos,
                              w_overload=args.w_overload, u_th=args.u_th,
                              static_power_ratio=args.static_power)
    tcn = TcnForecaster(REPO / args.tcn_ckpt if args.tcn_ckpt else DEFAULT_CKPT, device=args.device)
    env = CampusRLEnv(tcn, workdir=f"{args.out}/episodes",
                      minutes=args.minutes, scale=args.scale, sinr_threshold=args.sinr_threshold,
                      reward_cfg=reward_cfg, seed=args.seed, quiet=not args.verbose_ns3,
                      include_z=args.include_z, z_stats_path=args.z_stats)
    state_dim = env.observation_space["state"].shape[1]
    nbr = neighbor_matrix(env, env.observation_space["state"].shape[0])
    model = SharedCellActorCritic(state_dim, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    val_days = split_days("val", args.val_days)
    log = (out / "metrics.jsonl").open("a", encoding="utf-8")
    writer = make_writer(out / "tb", not args.no_tensorboard)
    if writer is not None:
        writer.add_text("config", "```\n" + json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n```")
    progress = Progress(not args.no_progress)
    steps_per_day = math.ceil(args.minutes / env.env.control_period)
    buffer_steps = steps_per_day * args.episodes_per_update
    print(f"[setup] state={state_dim}, params={sum(p.numel() for p in model.parameters())}, "
          f"device={device}, 하루 {steps_per_day} step, 버퍼 {buffer_steps} step, "
          f"val 날={len(val_days)}, 출력={out}", flush=True)

    try:
        from stub_policy import constant_max
        base = mean_summary(evaluate_days(env, lambda o: (constant_max(o), None), val_days,
                                          progress, "기준선(상시 최대) val"))
        print(f"[baseline] 상시 최대 val: {fmt(base)}", flush=True)
        log.write(json.dumps({"update": 0, "val_baseline_max": base}) + "\n")
        log_scalars(writer, "val_baseline_max", base, 0)
        env_steps = 0
        episode_count = 0
        best = -math.inf
        sample_fn = make_policy_fn(model, device, deterministic=False)
        greedy_fn = make_policy_fn(model, device, deterministic=True)
        for update in range(1, args.updates + 1):
            t0 = time.time()
            lr = args.lr if args.no_lr_anneal else args.lr * (1.0 - (update - 1) / args.updates)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = {k: [] for k in ("state", "action", "logp", "adv", "ret")}
            episodes = []
            model.eval()
            bar = progress.bar(buffer_steps, f"update {update}/{args.updates} 버퍼", "step")
            for _ in range(args.episodes_per_update):
                summary, traj = run_episode(env, sample_fn, collect=True,
                                            on_step=(lambda: bar.update(1)) if bar else None)
                episodes.append(summary)
                episode_count += 1
                env_steps += summary["steps"]
                log_scalars(writer, "episode", {k: summary[k] for k in
                                                ("reward", "saving_pct", "drop_pct", "overload_pct")},
                            episode_count)
                rewards = spatial_mix(np.stack(traj["reward"]), nbr, args.spatial_alpha) * args.reward_scale
                adv, ret = gae(rewards, np.stack(traj["value"]), args.gamma, args.lam)
                for k in ("state", "action", "logp"):
                    batch[k].extend(traj[k])
                batch["adv"].append(adv)
                batch["ret"].append(ret)
            if bar is not None:
                bar.close()
            model.train()
            stats = ppo_update(model, optimizer, batch, args, device)
            train = mean_summary(episodes)
            rec = {"update": update, "lr": lr, "train": train, "loss": stats,
                   "episodes": [(e["scenario"], e["day_seed"]) for e in episodes],
                   "seconds": time.time() - t0}
            log_scalars(writer, "train", train, update)
            log_scalars(writer, "loss", {"actor": stats["policy_loss"],
                                         "critic": stats["value_loss"],
                                         "entropy_per_cell": stats["entropy"],
                                         "approx_kl_per_cell": stats["kl"],
                                         "clip_fraction": stats["clip_frac"]}, update)
            log_scalars(writer, "progress", {"env_steps": env_steps, "episodes": episode_count,
                                             "lr": lr, "update_seconds": rec["seconds"]}, update)
            msg = (f"[update {update:4d}] 학습 {fmt(train)} | 엔트로피/셀 {stats['entropy']:.3f} "
                   f"| clip {stats['clip_frac']:.2f} | {rec['seconds']:.0f}s")
            if update % args.val_every == 0 or update == args.updates:
                model.eval()
                val = mean_summary(evaluate_days(env, greedy_fn, val_days, progress,
                                                 f"update {update} val"))
                rec["val"] = val
                log_scalars(writer, "val", val, update)
                log_scalars(writer, "val", {"reward_minus_max_baseline": val["reward"] - base["reward"]},
                            update)
                msg += f"\n             val  {fmt(val)}  (상시 최대 보상 {base['reward']:.3f})"
                if val["reward"] > best:
                    best = val["reward"]
                    torch.save({"model": model.state_dict(), "state_dim": state_dim,
                                "hidden": args.hidden, "update": update, "val": val,
                                "args": vars(args)}, out / "best.pt")
                    msg += "  -> best.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "state_dim": state_dim, "hidden": args.hidden, "update": update,
                        "args": vars(args)}, out / "last.pt")
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            if writer is not None:
                writer.flush()
            print(msg, flush=True)
    finally:
        log.close()
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
