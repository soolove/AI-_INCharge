"""Record one test day per 5-min step for the dashboard (agent PPO r10 or a baseline).

  python collect_day.py <repo> <out.npz> <max|reactive|rl> <scenario> <day_seed> [env label]

Writes <out.npz>, <out>.wp.csv (UE waypoints of the run), <out>.sites.csv and meta.json next to it.
Runs the ns-3 RL environment of <repo> with its ns-3 defaults (in this repository: the e-Um 5G radio
setting) on the CPU; each run uses its own ns-3 work folder, so runs can go in parallel. An env label, if
given, is stored in meta.json and shown by the render scripts.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

repo = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
method, scenario, day_seed = sys.argv[3], sys.argv[4], int(sys.argv[5])
env_label = sys.argv[6] if len(sys.argv) > 6 else ""
sys.path.insert(0, str(repo / "rl-env_v4"))
torch.set_num_threads(1)

from campus_env import CampusRLEnv  # noqa: E402
from ns3_antenna_env import RewardConfig  # noqa: E402
from policy import SharedCellActorCritic  # noqa: E402
from stub_policy import constant_max, threshold  # noqa: E402
from tcn_forecaster import TcnForecaster  # noqa: E402
from train_ppo import make_policy_fn  # noqa: E402

ck = torch.load(repo / "rl-env_v4/runs/ppo_v2_w64_r10/best.pt", map_location="cpu", weights_only=False)
cfg = ck["args"]
reward = RewardConfig(w_energy=cfg["w_energy"], w_qos=cfg["w_qos"], w_overload=cfg["w_overload"],
                      u_th=cfg.get("u_th", 0.8), static_power_ratio=cfg.get("static_power", 0.3))
tuned = json.loads((repo / "rl-env_v4/runs/reactive_tuned_v2_r10.json").read_text())
if method == "max":
    policy = lambda o: constant_max(o)  # noqa: E731
elif method == "reactive":
    policy = lambda o: threshold(o, up=tuned["up"], down=tuned["down"])  # noqa: E731
else:
    model = SharedCellActorCritic(ck["state_dim"], hidden=ck["hidden"])
    model.load_state_dict(ck["model"])
    model.eval()
    fn = make_policy_fn(model, torch.device("cpu"), deterministic=True)
    policy = lambda o: fn(o)[0]  # noqa: E731

from campus_env import SCENARIOS  # noqa: E402
workdir = f"rl-env_v4/run/demo_prev_{method}_s{SCENARIOS.index(scenario)}"   # one per run, so runs can go in parallel
env = CampusRLEnv(TcnForecaster(device="cpu"), workdir=workdir, minutes=cfg.get("minutes", 720),
                  reward_cfg=reward, include_z=cfg.get("include_z", False))
obs, info = env.reset(options={"scenario": scenario, "day_seed": day_seed})
rec = {k: [] for k in ("minute", "antenna", "traffic_min", "forecast", "ue", "outage_ue", "ho_in", "ho_out",
                       "shortage", "demand", "energy")}
while True:
    action = policy(obs)
    rec["forecast"].append(obs["forecast"])           # central t+1..t+15 and 15-min peak, made at this step
    obs, r, term, trunc, info = env.step(action)
    raw = obs["raw"]
    rec["minute"].append(float(obs["minute"][0]))
    rec["antenna"].append(raw[:, 7].copy())
    rec["traffic_min"].append(obs["tcn_window"][:, 0, -info["minutes"]:].copy())  # per-minute demand, last step
    rec["ue"].append(raw[:, 1].copy())
    rec["outage_ue"].append(raw[:, 8].copy())
    rec["ho_in"].append(np.asarray(info["ho_in_sum"]))
    rec["ho_out"].append(np.asarray(info["ho_out_sum"]))
    rec["shortage"].append(np.asarray(info["shortage_mbps"]))
    rec["demand"].append(np.asarray(info["demand_mbps"]))
    rec["energy"].append(info["energy"])
    if term or trunc:
        break
env.close()
np.savez(out, **{k: np.asarray(v) for k, v in rec.items()})
shutil.copy(repo / workdir / "wp_episode.csv", out.with_suffix(".wp.csv"))
shutil.copy(repo / workdir / "sites.csv", out.with_suffix(".sites.csv"))
day_meta = {"scenario": scenario, "day_seed": day_seed}
if env_label:
    day_meta["env"] = env_label
(out.parent / "meta.json").write_text(json.dumps(day_meta, ensure_ascii=False))
sh, dm = np.sum(rec["shortage"]), np.sum(rec["demand"])
print(f"{method}: steps {len(rec['minute'])}, saving {100 * (1 - np.mean(rec['energy'])):.1f}%, "
      f"QoS shortage {100 * sh / dm:.2f}%", flush=True)
