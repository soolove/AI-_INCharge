"""학습된 에이전트를 비교 기준(상시 최대, 반응형)과 같은 날에서 비교한다 (제안서 KPI 보고용).

표 다음에 KPI 요약을 낸다: KPI2 = 상시 최대 대비 에너지 절감률(목표 30% 이상), KPI3 = 반응형 제어 대비
QoS 부족(미처리 트래픽 = (셀 용량을 넘은 수요 + 끊긴 UE 수요) / 요구 트래픽) 감소율(목표 50% 이상).
끊김률은 그 순간의 판정이라 KPI로 쓰지 않고 참고로만 함께 보인다.

test 날은 train(시드 0~99999)·val(100000~)과 겹치지 않는 시드 200000~ 범위이며,
시나리오마다 --days개다. 같은 날에 모든 기법을 돌리므로 비교가 공정하다.
반응형 비교군은 tune_reactive.py가 val 날에서 고른 임계값을 쓴다.

  python rl-env_v4/evaluate.py --ckpt rl-env_v4/runs/ppo_v2_w64_r10/best.pt --days 5 \
      --reactive rl-env_v4/runs/reactive_tuned_v2_r10.json --workers 8

에이전트만 평가하려면 --policies rl (비교 기준 결과는 같은 날이면 매번 같으므로 이전 결과를 쓴다). --out이
없으면 일부 기법만 돌린 결과는 eval_<split>_rl.json처럼 따로 저장해 전체 결과 파일을 덮어쓰지 않는다.

--workers N: (기법, 날) 하루짜리 시뮬레이션들을 N개 프로세스가 나눠 동시에 돌린다. 각 프로세스는
자기 작업 폴더(<workdir>/w<번호>)와 공유 메모리 이름을 쓰고, 결과는 날 순서대로 합쳐 순서대로 돌린
것과 같은 숫자가 나온다 (ISAC 오차 난수를 날마다 정하고, 신경망은 CPU 1스레드로 돌리므로). ns-3와 Python이
공유 메모리에서 서로를 바쁜 대기로 기다려 프로세스 하나가 스레드 2개를 쓰므로 서버(16스레드)에서는 8 정도가
알맞다. 진행 막대(stderr)는 전체, 기법 x 시나리오별 완료 날 수, 프로세스별 현재 날과 진행 step을
보여준다 (tqdm이 없거나 터미널이 아니면 주기적인 글 요약).

끊김은 매분 관측 시점 기준(drop_pct)과 함께, 1분 사이를 --outage-samples번 나눠 본 기준으로도
센다(기본 60 = 1초마다): 끊김 시간 비율(drop_sampled_pct), 하루 끊김 횟수(outage_events),
그중 --outage-event-sec초(기본 10) 이상 이어진 횟수(outage_events_long). 이 샘플은 관측·보상을
바꾸지 않으므로 나머지 지표는 샘플 수와 무관하다.
"""

import argparse
import atexit
import json
import math
import multiprocessing as mp
import re
import signal
import sys
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campus_env import CampusRLEnv, SCENARIOS, split_days  # noqa: E402
from ns3_antenna_env import RewardConfig  # noqa: E402
from policy import SharedCellActorCritic  # noqa: E402
from stub_policy import constant_max, threshold  # noqa: E402
from tcn_forecaster import DEFAULT_CKPT, TcnForecaster  # noqa: E402
from train_ppo import make_policy_fn, mean_summary, run_episode  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
POLICIES_DEFAULT = "max,reactive,rl"
POLICY_ORDER = ("max", "reactive", "rl")
POLICY_NAMES = {"max": "상시 최대", "reactive": "반응형", "rl": "RL (PPO)"}
CONTROL_PERIOD = 5  # CampusRLEnv 기본 제어 주기(분) = run_episode의 step 하나
# 제안서 KPI 목표: KPI2 = 상시 최대 대비 에너지 절감률(%), KPI3 = 반응형 제어 대비 QoS 부족 감소율(%)
KPI2_TARGET_PCT = 30.0
KPI3_TARGET_PCT = 50.0


def make_env(setup, workdir):
    cfg = setup["cfg"]
    # 에이전트가 학습한 TCN과 z 통계로 평가한다(train_ppo.py --tcn-ckpt/--z-stats). 체크포인트에 없거나
    # 비교 기준만 평가하면 제공된 TCN.
    tcn = cfg.get("tcn_ckpt")
    return CampusRLEnv(TcnForecaster(REPO / tcn if tcn else DEFAULT_CKPT, device=setup["device"]),
                       workdir=workdir,
                       minutes=cfg.get("minutes", 720), scale=cfg.get("scale", 1.0),
                       sinr_threshold=cfg.get("sinr_threshold", -5.0),
                       reward_cfg=RewardConfig(**setup["reward_cfg"]), quiet=setup["quiet"],
                       include_z=cfg.get("include_z", False),
                       outage_samples_per_min=setup["outage_samples"],
                       outage_event_sec=setup["outage_event_sec"], ns3_args=setup["ns3_args"],
                       z_stats_path=cfg.get("z_stats"))


def make_policies(setup):
    up, down = setup["up"], setup["down"]
    policies = {"max": lambda o: (constant_max(o), None),
                "reactive": lambda o: (threshold(o, up=up, down=down), None)}
    if "rl" in setup["keys"]:
        ck = torch.load(setup["ckpt"], map_location="cpu", weights_only=False)
        model = SharedCellActorCritic(ck["state_dim"], hidden=ck["hidden"]).to(setup["device"])
        model.load_state_dict(ck["model"])
        model.eval()
        policies["rl"] = make_policy_fn(model, torch.device(setup["device"]), deterministic=True)
    return policies


# ---------------------------------------------------------------- 병렬 실행 (작업 프로세스 쪽)
_WORKER = {}


def _worker_init(setup, worker_ids, progress):
    """작업 프로세스마다 한 번: 기법별 행동 함수를 만든다 (환경은 첫 작업 때).

    Ctrl+C와 터미널 끊김은 메인이 막은 채로 이 프로세스를 띄워(run_tasks) 모든 스레드에서 막혀 있다.
    """
    torch.set_num_threads(1)
    _WORKER.update(setup=setup, policies=make_policies(setup), wid=worker_ids.get(),
                   progress=progress, env=None)


def _worker_run(task):
    key, scenario, day_seed = task
    w = _WORKER
    w["progress"].put(("start", w["wid"], task))
    try:
        if w["env"] is None:
            # ns-3를 처음 띄우는 것도 'start' 뒤에 해 진행 감시(--stall-timeout) 안에 둔다. 작업을 받지 못한
            # 프로세스는 ns-3를 띄우지 않으므로 끝낼 때 기다릴 것이 없다.
            w["env"] = make_env(w["setup"], f"{w['setup']['workdir']}/w{w['wid']}")
            atexit.register(w["env"].close)  # 정상 종료 때 ns-3 프로세스와 공유 메모리를 정리한다
        row = run_episode(w["env"], w["policies"][key], {"scenario": scenario, "day_seed": day_seed},
                          on_step=lambda: w["progress"].put(("step", w["wid"], task)))[0]
    except (Exception, SystemExit) as e:
        # ns3-ai는 ns-3가 바로 죽으면 exit(1)을 부른다: 메인에서 조용히 끝나지 않게 어느 날인지 붙여 올린다.
        raise RuntimeError(f"w{w['wid']} {POLICY_NAMES[key]} | {scenario} {day_seed} 실패") from e
    w["progress"].put(("done", w["wid"], task))
    return task, row


# ---------------------------------------------------------------- 진행 표시 (메인 프로세스)
def _disp_width(text):
    """터미널 표시 폭 (한글 같은 전각 문자는 2칸)."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def _pad(text, width):
    """표시 폭 기준으로 오른쪽을 공백으로 채운다."""
    return text + " " * max(0, width - _disp_width(text))


class Progress:
    """전체 / 기법 x 시나리오별 완료 날 수 / 프로세스별 현재 날과 step. stderr에 그린다."""

    def __init__(self, tasks, n_workers, steps_per_day, enabled=True):
        self.total = len(tasks)
        self.steps_per_day = steps_per_day
        self.done = 0
        self.t0 = time.time()
        self.groups = {}
        for key, scenario, _ in tasks:
            self.groups.setdefault((key, scenario), [0, 0])[1] += 1
        self.worker_task = {w: None for w in range(n_workers)}
        self.worker_step = {w: 0 for w in range(n_workers)}
        self.last_text = 0.0
        self.last_done = -1  # 글 요약을 마지막으로 찍었을 때의 완료 날 수
        if not enabled:
            self.mode = "off"
            return
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        self.mode = "bars" if (tqdm is not None and sys.stderr.isatty()) else "text"
        if self.mode != "bars":
            return
        # 설명 칸 폭을 모두 같게 해 막대 시작을 맞춘다 (프로세스 줄은 날이 바뀌어도 같은 폭).
        labels = ["전체"] + [self._group_label(g) for g in self.groups]
        labels += [self._worker_label(n_workers - 1, t) for t in tasks]
        self.width = max(_disp_width(x) for x in labels)
        days = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}일"
        kw = dict(file=sys.stderr, dynamic_ncols=True, leave=True, mininterval=0.3)
        pos = 0
        self.bar_all = tqdm(total=self.total, desc=_pad("전체", self.width), position=pos,
                            bar_format=days + " [{elapsed}<{remaining}]", **kw)
        self.bar_group = {}
        for g, (_, n) in self.groups.items():
            pos += 1
            self.bar_group[g] = tqdm(total=n, desc=_pad(self._group_label(g), self.width),
                                     position=pos, bar_format=days, **kw)
        self.bar_worker = {}
        for w in range(n_workers):
            pos += 1
            self.bar_worker[w] = tqdm(
                total=steps_per_day, desc=_pad(f"  w{w} 대기", self.width), position=pos,
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} step [{elapsed}<{remaining}]",
                **kw)

    @staticmethod
    def _group_label(group):
        return f"{POLICY_NAMES[group[0]]} | {group[1]}"

    @staticmethod
    def _worker_label(wid, task):
        key, scenario, day_seed = task
        return f"  w{wid} {POLICY_NAMES[key]} | {scenario} {day_seed}"

    def handle(self, msg):
        kind, wid, task = msg
        key, scenario, day_seed = task
        if kind == "start":
            self.worker_task[wid] = task
            self.worker_step[wid] = 0
            if self.mode == "bars":
                bar = self.bar_worker[wid]
                bar.reset(total=self.steps_per_day)
                bar.set_description_str(_pad(self._worker_label(wid, task), self.width))
        elif kind == "step":
            self.worker_step[wid] += 1
            if self.mode == "bars":
                self.bar_worker[wid].update(1)
        elif kind == "done":
            self.done += 1
            self.groups[(key, scenario)][0] += 1
            self.worker_task[wid] = None
            if self.mode == "bars":
                self.bar_all.update(1)
                self.bar_group[(key, scenario)].update(1)
        if self.mode == "text" and (kind == "done" or time.time() - self.last_text > 15.0):
            self._print_text()

    def _print_text(self):
        self.last_text = time.time()
        self.last_done = self.done
        groups = " | ".join(f"{POLICY_NAMES[k]} {s} {d}/{n}" for (k, s), (d, n) in self.groups.items())
        busy = [f"w{w} {POLICY_NAMES[t[0]]} {t[1]} {t[2]} {self.worker_step[w]}/{self.steps_per_day}"
                for w, t in self.worker_task.items() if t is not None]
        print(f"[진행 {self.done}/{self.total}일, {time.time() - self.t0:.0f}s] {groups}"
              + (f"\n    진행 중: {', '.join(busy)}" if busy else ""), file=sys.stderr, flush=True)

    def close(self):
        if self.mode == "bars":
            for bar in [self.bar_all, *self.bar_group.values(), *self.bar_worker.values()]:
                bar.close()
        elif self.mode == "text" and self.last_done != self.done:
            self._print_text()


_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def _ignore_stop_signals():
    """곧 끝나는 프로세스의 정리 직전에: 두 번째 Ctrl+C 등이 정리를 끊지 못하게 한다."""
    for sig in _STOP_SIGNALS:
        signal.signal(sig, signal.SIG_IGN)


def _kill_pool(ex, worker_pids):
    """중단·오류 때: 작업 프로세스와 그 ns-3(실행기 포함)를 모두 끝내고 공유 메모리를 지운다.

    ns-3는 따로 프로세스 그룹을 만들고, 작업 프로세스 하나가 죽으면 실행기가 나머지를 SIGTERM으로
    끝내 그 ns-3들이 부모 없이 남는다 (응답을 기다리며 CPU를 계속 쓴다). 그래서 자식 관계가 아니라
    실행 인자의 공유 메모리 이름(--rlShmTag=pid<작업 프로세스>)으로 찾는다.
    """
    import psutil  # ns3ai_utils가 쓰는 패키지 (같은 환경에 설치돼 있음)

    ex.shutdown(wait=False, cancel_futures=True)  # 남은 작업을 더 넘기지 않게 먼저 멈춘다
    victims = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        tags = re.findall(r"--rlShmTag=pid(\d+)", " ".join(proc.info["cmdline"] or []))
        if proc.info["pid"] in worker_pids or any(int(t) in worker_pids for t in tags):
            victims.append(proc)
    for proc in victims:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(victims, timeout=10)
    for pid in worker_pids:  # campus_env.shm_names(f"pid{pid}")의 세그먼트
        for seg in Path("/dev/shm").glob(f"ns3ai * pid{pid}"):
            seg.unlink(missing_ok=True)


def run_tasks(setup, tasks, workers, progress, stall_timeout):
    """(기법, 시나리오, 날) 작업을 모두 돌려 {작업: 하루 요약}을 돌려준다."""
    rows = {}
    if workers <= 1:
        env = make_env(setup, setup["workdir"])
        policies = make_policies(setup)
        try:
            for task in tasks:
                key, scenario, day_seed = task
                progress.handle(("start", 0, task))
                rows[task] = run_episode(env, policies[key], {"scenario": scenario, "day_seed": day_seed},
                                         on_step=lambda t=task: progress.handle(("step", 0, t)))[0]
                progress.handle(("done", 0, task))
        except BaseException:
            _ignore_stop_signals()
            env.close()
            raise
        env.close()
        return rows

    ctx = mp.get_context("spawn")  # CUDA·ns3-ai 상태를 물려받지 않는 새 프로세스
    last_seen = {}  # 작업 프로세스 -> 마지막 진행 소식 (time.monotonic, 날을 돌리는 중일 때만)
    ex = None
    worker_pids = set()
    # Ctrl+C와 터미널 끊김은 같은 프로세스 그룹의 자식에게도 가지만 정리는 메인이 한다. 자식은 띄울 때의 신호
    # 마스크를 물려받으므로, 보조 프로세스(첫 Queue가 띄우는 resource tracker)와 작업 프로세스(submit 때 뜸)를
    # 막아 둔 채 띄우면 그 안의 모든 스레드(import 때 생기는 numpy 스레드 포함)가 막힌 채 시작한다.
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGHUP})
    try:
        try:
            worker_ids = ctx.Queue()
            for w in range(workers):
                worker_ids.put(w)
            # 진행 소식은 보내는 스레드가 바로 쓰는 SimpleQueue로 받는다. Queue는 보조 스레드가 프로세스 사이 잠금을
            # 쥔 채 자기 프로세스의 GIL을 기다리는데, 작업 프로세스는 GIL을 쥔 채 ns-3 응답을 바쁜 대기로 기다린다.
            # 그때 ns-3가 죽으면 잠금이 영영 풀리지 않아 모든 소식이 멈추고, 감시가 엉뚱한 프로세스를 지목한다.
            messages = ctx.SimpleQueue()
            ex = ProcessPoolExecutor(workers, mp_context=ctx, initializer=_worker_init,
                                     initargs=(setup, worker_ids, messages))
            pending = {ex.submit(_worker_run, task) for task in tasks}
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        worker_pids |= set(ex._processes)  # submit 때 모두 뜨고 바뀌지 않는다 (죽은 뒤에도 찾을 수 있게 기록)

        def drain():
            while not messages.empty():
                msg = messages.get()
                kind, wid, _ = msg
                if kind == "done":
                    last_seen.pop(wid, None)
                else:
                    last_seen[wid] = time.monotonic()
                progress.handle(msg)

        last_loop = time.monotonic()
        while pending:
            done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            now = time.monotonic()
            if now - last_loop > 5.0:  # 메인이 멈춰 있던 시간(Ctrl+Z 등)은 진행 없음으로 치지 않는다
                for w in last_seen:
                    last_seen[w] += now - last_loop
            last_loop = now
            drain()
            for fut in done:
                task, row = fut.result()  # 작업 프로세스의 예외는 여기서 다시 올라온다
                rows[task] = row
            # ns-3가 죽으면 그 작업 프로세스는 응답을 끝없이 기다린다: 진행이 멈추면 중단한다.
            stalled = [w for w, t in last_seen.items() if now - t > stall_timeout]
            if stalled:
                w = stalled[0]
                key, scenario, day_seed = progress.worker_task[w]
                raise RuntimeError(f"작업 프로세스 w{w}({POLICY_NAMES[key]} | {scenario} {day_seed})가 "
                                   f"{stall_timeout:.0f}초 동안 진행이 없습니다 (ns-3 프로세스가 비정상 종료됐을 "
                                   "수 있음). --verbose-ns3로 원인을 확인하세요")
        time.sleep(0.2)
        drain()
        ex.shutdown(wait=True)
    except BaseException:
        _ignore_stop_signals()
        if ex is not None:
            _kill_pool(ex, worker_pids | set(getattr(ex, "_processes", None) or {}))
        raise
    return rows


def _reduction(base, agent):
    """감소율(%) = 1 - 에이전트/비교 기준. 기준이 0이면 nan."""
    return 100.0 * (1.0 - agent / base) if base > 0 else float("nan")


def kpi_summary(results, days):
    """에이전트의 KPI2(절감률)와 KPI3(반응형 대비 QoS 부족 감소율). 끊김률은 참고로 함께 담는다.

    QoS 부족 = (셀 용량을 넘은 수요 + 끊긴 UE의 수요) / 요구 트래픽이라, 끊긴 사용자의 수요도 들어간다.
    """
    agent = results.get(POLICY_NAMES["rl"])
    if agent is None:
        return {}
    kpi = {"kpi2": {"saving_pct": agent["all"]["saving_pct"], "target_pct": KPI2_TARGET_PCT}}
    react = results.get(POLICY_NAMES["reactive"])
    if react is not None:
        kpi["kpi3"] = {
            "reactive_shortage_pct": react["all"]["shortage_pct"],
            "agent_shortage_pct": agent["all"]["shortage_pct"],
            "reduction_pct": _reduction(react["all"]["shortage_pct"], agent["all"]["shortage_pct"]),
            "target_pct": KPI3_TARGET_PCT,
            "reduction_by_scenario_pct": {sc: _reduction(react[sc]["shortage_pct"], agent[sc]["shortage_pct"])
                                          for sc in SCENARIOS},
            "days_agent_lower": sum(a["shortage_pct"] < r["shortage_pct"]
                                    for r, a in zip(react["days"], agent["days"])),
            "days": len(days),
            "outage_ref": {"reactive_drop_sampled_pct": react["all"]["drop_sampled_pct"],
                           "agent_drop_sampled_pct": agent["all"]["drop_sampled_pct"]},
        }
    return kpi


def print_kpi(kpi, split, n_days, step_sec):
    if not kpi:
        return
    k2 = kpi["kpi2"]
    print(f"KPI (에이전트, {split} {n_days}일 평균)")
    print(f"  KPI2 에너지 절감률 (상시 최대 대비): {k2['saving_pct']:.1f}%  "
          f"(목표 {k2['target_pct']:g}% 이상: {'달성' if k2['saving_pct'] >= k2['target_pct'] else '미달'})")
    k3 = kpi.get("kpi3")
    if k3 is not None:
        print(f"  KPI3 QoS 부족 (반응형 대비): {k3['reactive_shortage_pct']:.2f}% -> {k3['agent_shortage_pct']:.2f}%, "
              f"{k3['reduction_pct']:.1f}% 감소  (목표 {k3['target_pct']:g}% 이상: "
              f"{'달성' if k3['reduction_pct'] >= k3['target_pct'] else '미달'})")
        by_sc = ", ".join(f"{sc} {v:.1f}%" for sc, v in k3["reduction_by_scenario_pct"].items())
        print(f"       시나리오별 감소: {by_sc} | 날별 {k3['days_agent_lower']}/{k3['days']}일 에이전트가 낮음")
        o = k3["outage_ref"]
        print(f"  참고 끊김률({step_sec:g}초 판정): 반응형 {o['reactive_drop_sampled_pct']:.2f}%, "
              f"에이전트 {o['agent_drop_sampled_pct']:.2f}%")
    print()


def _exit_on_signal(signum, frame):
    raise SystemExit(128 + signum)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="학습된 에이전트 체크포인트 (없으면 비교 기준만)")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--days", type=int, default=5, help="시나리오당 날 수")
    ap.add_argument("--reactive", default="rl-env_v4/runs/reactive_tuned_v2_r10.json",
                    help="tune_reactive.py 결과 (없으면 stub_policy 기본값 up=0.03, down=0.02)")
    ap.add_argument("--outage-samples", type=int, default=60,
                    help="끊김을 1분에 몇 번 볼지 (60 = 1초마다, 1 = 매분 관측 시점만)")
    ap.add_argument("--outage-event-sec", type=float, default=10.0,
                    help="이 시간(초) 이상 이어진 끊김을 긴 끊김으로 따로 센다")
    ap.add_argument("--out", default=None, help="결과 JSON 경로 (기본: 체크포인트 옆)")
    ap.add_argument("--policies", default=POLICIES_DEFAULT,
                    help="평가할 기법 (쉼표로): max = 상시 최대, reactive = 반응형 (둘은 비교 기준), "
                         "rl = 학습된 에이전트. 에이전트만 보려면 rl (반응형 조정 파일은 읽지 않음). "
                         "일부만 돌리면 --out이 없을 때 결과를 eval_<split>_<기법>.json에 따로 저장")
    ap.add_argument("--workers", type=int, default=1,
                    help="동시에 돌릴 프로세스 수 (1 = 순서대로). 서버(16스레드)는 8 정도")
    ap.add_argument("--workdir", default="rl-env_v4/run/eval",
                    help="에피소드 입력 파일 작업 폴더. 평가 여러 개를 동시에 돌리려면 서로 다르게 준다")
    ap.add_argument("--device", default="cpu",
                    help="TCN·에이전트 신경망 장치. 기본 CPU(1스레드): 결과가 프로세스 수와 무관하게 같고 "
                         "병렬 프로세스가 GPU를 다투지 않는다")
    ap.add_argument("--stall-timeout", type=float, default=300.0,
                    help="병렬 실행에서 한 프로세스가 이 시간(초) 동안 진행이 없으면 중단 (ns-3 이상 감지)")
    ap.add_argument("--no-progress", action="store_true", help="진행 표시 끄기")
    ap.add_argument("--verbose-ns3", action="store_true", help="ns3-ai 실행 메시지 보여주기")
    ap.add_argument("--ns3-arg", action="append", default=[], metavar="이름=값",
                    help="ns-3 실행 인자를 더하거나 덮어쓴다 (여러 번 가능). 예: v2 무선 설정(3.5 GHz, "
                         "침투 손실 없음) = --ns3-arg slmCarrierFrequency=3.5e9 --ns3-arg o2iLossDb=0")
    args = ap.parse_args()
    if args.workers < 1 or args.days < 1 or args.outage_samples < 1:
        ap.error("--workers, --days, --outage-samples는 1 이상")
    if any("=" not in a for a in args.ns3_arg):
        ap.error("--ns3-arg는 이름=값 형식")
    ns3_args = dict(a.split("=", 1) for a in args.ns3_arg)
    # ns3-ai는 환경을 만들 때 작업 디렉터리를 저장소로 바꾼다 (프로세스 1개면 메인에서). 상대 경로가 그 전후로
    # 다르게 풀리지 않도록 미리 절대 경로로 바꿔 둔다.
    args.ckpt = str(Path(args.ckpt).resolve()) if args.ckpt else None
    args.out = str(Path(args.out).resolve()) if args.out else None

    wanted = {p.strip() for p in args.policies.split(",") if p.strip()}
    if not wanted or not wanted <= set(POLICY_ORDER):
        ap.error("--policies는 max, reactive, rl 중에서 쉼표로 고른다")
    if "rl" in wanted and not args.ckpt:
        if args.policies != POLICIES_DEFAULT:
            ap.error("--policies에 rl이 있으면 --ckpt가 필요하다")
        wanted.discard("rl")  # 체크포인트 없이 기본값이면 비교 기준만 (이전 동작)
    keys = [k for k in POLICY_ORDER if k in wanted]
    torch.set_num_threads(1)  # 순서대로 돌려도 병렬과 같은 숫자가 나오도록

    tuned = REPO / args.reactive
    up = down = None
    if "reactive" in wanted:
        if tuned.exists():
            r = json.loads(tuned.read_text())
            up, down = r["up"], r["down"]
            print(f"반응형 임계값: up={up}, down={down} (val 조정값, {tuned})")
        else:
            up, down = 0.03, 0.02
            print(f"[경고] {tuned}가 없어 stub_policy 기본 임계값 up=0.03, down=0.02를 씁니다")

    cfg = {}
    if "rl" in wanted:
        cfg = torch.load(args.ckpt, map_location="cpu", weights_only=False)["args"]
    # 학습 때와 같은 보상·끊김 설정으로 평가해야 숫자가 비교 가능하다. 체크포인트가 없으면 기본값(r10 설정).
    base = RewardConfig()
    reward_cfg = RewardConfig(w_energy=cfg.get("w_energy", base.w_energy), w_qos=cfg.get("w_qos", base.w_qos),
                              w_overload=cfg.get("w_overload", base.w_overload),
                              u_th=cfg.get("u_th", base.u_th),
                              static_power_ratio=cfg.get("static_power", base.static_power_ratio))
    # 반응형 임계값은 보상 기준으로 고른 값이라, 보상 가중치가 다른 조정 결과를 쓰면 비교가 틀어진다.
    if "reactive" in wanted and tuned.exists() and vars(reward_cfg) != r["reward_config"]:
        sys.exit(f"[중단] {tuned}의 보상 설정 {r['reward_config']}이 평가 설정 {vars(reward_cfg)}과 다릅니다. "
                 "같은 가중치로 tune_reactive.py를 돌린 파일을 --reactive로 넘기세요")

    setup = {"keys": keys, "ckpt": args.ckpt, "cfg": cfg, "reward_cfg": vars(reward_cfg),
             "up": up, "down": down, "device": args.device, "quiet": not args.verbose_ns3,
             "workdir": args.workdir, "outage_samples": args.outage_samples,
             "outage_event_sec": args.outage_event_sec, "ns3_args": ns3_args}
    days = split_days(args.split, args.days)
    tasks = [(key, scenario, day_seed) for key in keys for scenario, day_seed in days]
    workers = min(args.workers, len(tasks))
    steps_per_day = math.ceil(cfg.get("minutes", 720) / CONTROL_PERIOD)
    print(f"평가: {', '.join(POLICY_NAMES[k] for k in keys)} x {len(days)}일 = {len(tasks)}개, "
          f"프로세스 {workers}개, 끊김 판정 {60.0 / args.outage_samples:g}초 간격", flush=True)
    if ns3_args:
        print(f"ns-3 인자 추가: {ns3_args}", flush=True)
    progress = Progress(tasks, workers, steps_per_day, enabled=not args.no_progress)
    # kill이나 터미널 끊김에도 Ctrl+C처럼 작업 프로세스와 ns-3를 정리하고 끝낸다 (nohup으로 무시한 신호는 그대로).
    for sig in (signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(sig) is not signal.SIG_IGN:
            signal.signal(sig, _exit_on_signal)
    t0 = time.time()
    try:
        rows = run_tasks(setup, tasks, workers, progress, args.stall_timeout)
    except KeyboardInterrupt:
        sys.exit("\n[중단] Ctrl+C: 작업 프로세스와 ns-3를 정리하고 끝냈습니다")
    finally:
        progress.close()
    print(f"[완료] {len(tasks)}개, {time.time() - t0:.0f}초", flush=True)

    results = {}
    for key in keys:
        name = POLICY_NAMES[key]
        day_rows = [rows[(key, scenario, day_seed)] for scenario, day_seed in days]
        results[name] = {"all": mean_summary(day_rows), "days": day_rows}
        for sc in SCENARIOS:
            results[name][sc] = mean_summary([x for x in day_rows if x["scenario"] == sc])

    step_sec = 60.0 / args.outage_samples
    print(f"\n{args.split} 날: 시나리오 {len(SCENARIOS)}개 x {args.days}일 = {len(days)}일 "
          f"(보상은 셀 평균의 하루 합, 끊김률은 매분 관측 시점 기준)")
    long_note = ("" if step_sec < args.outage_event_sec
                 else " (판정 간격이 기준 이상이라 측정 불가: -)")
    print(f"분 사이 끊김: {step_sec:g}초마다 판정, 끊김 시간 비율 / 하루 끊김 횟수 / "
          f"{args.outage_event_sec:g}초 이상 이어진 하루 끊김 횟수{long_note}")
    print(f"{'기법':10s}{'구간':8s}{'보상':>9s}{'절감률':>9s}{'QoS 부족':>10s}{'끊김률':>9s}{'과부하':>9s}"
          f"{'끊김(초)':>10s}{'끊김 횟수':>10s}{'긴 끊김':>9s}")
    for name, res in results.items():
        for part in ("all",) + SCENARIOS:
            m = res[part]
            label = "전체" if part == "all" else part
            n_long = m["outage_events_long"]
            long_s = f"{n_long:9.1f}" if n_long == n_long else f"{'-':>9s}"
            print(f"{name:10s}{label:8s}{m['reward']:9.3f}{m['saving_pct']:8.1f}%"
                  f"{m['shortage_pct']:9.2f}%{m['drop_pct']:8.2f}%{m['overload_pct']:8.2f}%"
                  f"{m['drop_sampled_pct']:9.2f}%{m['outage_events']:10.1f}{long_s}")
        print()

    kpi = kpi_summary(results, days)
    print_kpi(kpi, args.split, len(days), step_sec)

    # 일부 기법만 돌린 결과가 전체 결과 파일을 덮어쓰지 않도록 이름을 나눈다.
    full = [k for k in POLICY_ORDER if k != "rl" or args.ckpt]
    suffix = "" if keys == full else "_" + "_".join(keys)
    out = Path(args.out) if args.out else (
        Path(args.ckpt).with_name(f"eval_{args.split}{suffix}.json") if args.ckpt
        else REPO / f"rl-env_v4/runs/baselines_{args.split}{suffix}.json")
    out.write_text(json.dumps({"reactive": {"up": up, "down": down},
                               "outage_sampling": {"samples_per_min": args.outage_samples,
                                                   "sample_sec": step_sec,
                                                   "long_event_sec": args.outage_event_sec},
                               "protocol": {"isac_noise": "per_scenario_day", "device": args.device,
                                            "torch_threads": 1},
                               "ns3_args": ns3_args,
                               "results": results,
                               "kpi": kpi},
                              ensure_ascii=False, indent=2))
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
