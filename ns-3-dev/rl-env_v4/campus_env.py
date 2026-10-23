"""RL 학습·평가용 캠퍼스 환경: 에피소드마다 (시나리오, 날)을 바꿔 ns-3를 다시 띄운다.

1 에피소드 = 시뮬레이션 하루(09:00~21:00, 5분 step x 144). 날은 시나리오 인원표에
일간 변동(build_tcn_dataset.py와 같은 day_var=0.12)을 줘서 만들고, 날의 정체는 정수
시드 하나로 정해진다. train / val / test는 시드 범위로 분리해 서로 겹치지 않는다.

ns3-ai의 Ns3Env.reset()은 ns-3를 다시 띄울 때 실행 인자를 넘기지 않아 시나리오를
바꿀 수 없다. ReconfigurableNs3Env가 이 부분만 고쳐, configure()로 바꾼 설정으로
다음 reset() 때 새 날을 띄운다. 공유 메모리 이름은 프로세스마다 달라 한 컴퓨터에서
여러 실험을 동시에 돌릴 수 있고, 한 프로세스에는 환경을 하나만 둔다. 학습과 검증은 같은
환경에서 reset 옵션으로 날을 고른다.
"""

import contextlib
import json
import io
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ns3ai_gym_env  # noqa: F401,E402  gym 환경 등록
from ns3ai_gym_env.envs.ns3_environment import Ns3Env  # noqa: E402
import ns3ai_gym_msg_py as py_binding  # noqa: E402
from ns3ai_utils import Experiment  # noqa: E402

import scenario_setup  # noqa: E402
from ns3_antenna_env import Ns3AntennaEnv, RewardConfig  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SCENARIOS = ("일반일과", "시험기간", "축제")
TRAIN_SEED_RANGE = (0, 100000)
SPLIT_SEED_BASE = {"val": 100000, "test": 200000}
Z_STATS = Path(__file__).resolve().parent / "z_stats.json"


def split_days(split, days_per_scenario):
    """val/test의 고정 날 목록 [(시나리오, 날 시드)]. 시나리오마다 같은 개수."""
    base = SPLIT_SEED_BASE[split]
    return [(sc, base + k) for sc in SCENARIOS for k in range(days_per_scenario)]


@contextlib.contextmanager
def _quiet(enabled=True):
    """ns3-ai가 에피소드마다 찍는 진행 메시지(Running ns-3 with ..., Killing subprocesses...)를
    감춘다. 도중에 예외나 종료가 일어나면 감춘 내용을 stderr로 다시 내보내 원인을 볼 수 있게 한다
    (ns3-ai는 ns-3가 바로 죽으면 'Subprocess died very early'를 출력하고 exit(1) 한다)."""
    if not enabled:
        yield
        return
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            yield
    except BaseException:
        sys.stderr.write(buf.getvalue())
        raise


def shm_names(tag):
    """ns3-ai 공유 메모리 이름. cttc-nr-demo.cc의 --rlShmTag 처리와 같은 형식이어야 한다."""
    return {"segName": f"ns3ai seg {tag}", "cpp2pyMsgName": f"ns3ai cpp2py {tag}",
            "py2cppMsgName": f"ns3ai py2cpp {tag}", "lockableName": f"ns3ai lockable {tag}"}


class ReconfigurableNs3Env(Ns3Env):
    """configure()로 바꾼 실행 인자를 다음 reset()에서 ns-3에 넘기는 Ns3Env.

    Ns3Env는 공유 메모리 이름을 기본값으로 고정해, 같은 컴퓨터의 두 프로세스가 서로의
    메모리를 지우고 덮어쓴다. 여기서는 프로세스 ID로 이름을 정하고 ns-3에도 --rlShmTag로
    같은 값을 넘겨 학습·조정·평가를 동시에 돌릴 수 있게 한다.
    """

    def __init__(self, targetName, ns3Path, ns3Settings, quiet=True):
        # Ns3Env.__init__과 같은 순서로 초기화하되 Experiment에 공유 메모리 이름을 넘긴다.
        self.quiet = quiet
        self.shm_tag = f"pid{os.getpid()}"
        self.ns3Settings = self._tagged(ns3Settings)
        self.newStateRx = False
        self.obsData = None
        self.reward = 0
        self.gameOver = False
        self.gameOverReason = None
        self.extraInfo = None
        with _quiet(quiet):
            self.exp = Experiment(targetName, ns3Path, py_binding, **shm_names(self.shm_tag))
            self.msgInterface = self.exp.run(setting=self.ns3Settings, show_output=True)
            self.initialize_env()
            self.rx_env_state()
        self.envDirty = False
        self._pending = False

    def _tagged(self, settings):
        return dict(settings, rlShmTag=self.shm_tag)

    def configure(self, settings):
        self.ns3Settings = self._tagged(settings)
        self._pending = True

    def reset(self, seed=None, options=None):
        if not self.envDirty and not self._pending:
            return self.get_obs(), {}
        if not self.gameOver:
            # 직전 상태는 이미 받았으므로(rx_env_state는 중복 수신하지 않음) 종료 명령만 보낸다.
            self.rx_env_state()
            self.send_close_command()
        # ns-3가 종료 명령을 읽고 스스로 끝날 때까지 기다린다. 읽기 전에 죽이면 명령이 공유 메모리에 남아, 새
        # ns-3가 그것을 첫 응답으로 읽고 바로 끝나 버린다 (Python은 첫 관측을 끝없이 기다림).
        if self.exp.proc is not None:
            try:
                self.exp.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass  # exp.run()이 남은 프로세스를 죽인다
        self.msgInterface = None
        self.newStateRx = False
        self.obsData = None
        self.reward = 0
        self.gameOver = False
        self.gameOverReason = None
        self.extraInfo = None
        with _quiet(self.quiet):
            self.msgInterface = self.exp.run(setting=self.ns3Settings, show_output=True)
        self.initialize_env()
        self.rx_env_state()
        self.envDirty = False
        self._pending = False
        return self.get_obs(), {}

    def close(self):
        with _quiet(self.quiet):
            super().close()


class CampusRLEnv(gym.Wrapper):
    """Ns3AntennaEnv 위에서 에피소드마다 (시나리오, 날)을 고른다.

    reset(options={"scenario": ..., "day_seed": ...})로 날을 지정할 수 있고, 지정하지 않으면
    train 시드 범위에서 무작위로 고른다. info에 이번 에피소드의 시나리오와 시드를 담는다.
    """

    def __init__(self, forecaster, workdir="rl-env_v4/run/episodes", minutes=720, scale=1.0,
                 day_var=0.12, control_period=5, sinr_threshold=-5.0,
                 reward_cfg=None, seed=0, quiet=True, include_z=True,
                 outage_samples_per_min=1, outage_event_sec=10.0, ns3_args=None, z_stats_path=None):
        if outage_samples_per_min < 1 or outage_event_sec < 0:
            raise ValueError("outage_samples_per_min >= 1, outage_event_sec >= 0 이어야 합니다")
        self.workdir = (REPO / workdir).resolve()
        self.minutes = minutes
        # ns-3 실행 인자를 더하거나 덮어쓴다 (예: v2 무선 설정 비교 {"slmCarrierFrequency": 3.5e9,
        # "o2iLossDb": 0}). 비우면 ns-3 기본값(이음5G 4.77 GHz, 실내 침투 손실 16.2 dB)을 쓴다.
        self.ns3_args = dict(ns3_args or {})
        # 분 사이 끊김 샘플 수(1 = 매분 관측 시점만, 60 = 1초마다)와 긴 끊김 기준(초). 평가 지표용이며
        # 관측·보상에는 영향이 없다.
        self.outage_samples_per_min = int(outage_samples_per_min)
        self.outage_event_sec = float(outage_event_sec)
        self.scale = scale
        self.day_var = day_var
        self.sinr_threshold = sinr_threshold
        self._rng = np.random.default_rng(seed)
        self._first_episode = self._sample_train()
        files = self._prepare(*self._first_episode)
        # --no-build: 학습 중 매 에피소드마다 빌드 확인을 하지 않는다(빌드는 미리 해 둘 것).
        self.ns3 = ReconfigurableNs3Env(targetName="--no-build cttc-nr-demo", ns3Path=str(REPO),
                                        ns3Settings=self._settings(files, self._first_episode[1]),
                                        quiet=quiet)
        # z 표준화 통계는 forecaster의 TCN에서 구한 것이어야 한다(compute_z_stats.py). 기본은 제공된 TCN의 것.
        z_stats_file = REPO / z_stats_path if z_stats_path else Z_STATS
        env = Ns3AntennaEnv(self.ns3, n_cells=files["n_cells"], control_period=control_period,
                            reward_cfg=reward_cfg or RewardConfig(), episode_minutes=minutes,
                            forecaster=forecaster, include_z=include_z,
                            z_stats=json.loads(z_stats_file.read_text()) if include_z else None,
                            seed=seed)
        super().__init__(env)
        self._started = False
        self.episode = None

    def _sample_train(self):
        scenario = SCENARIOS[int(self._rng.integers(len(SCENARIOS)))]
        return scenario, int(self._rng.integers(*TRAIN_SEED_RANGE))

    def _prepare(self, scenario, day_seed):
        return scenario_setup.prepare(self.workdir, scenario=scenario, minutes=self.minutes,
                                      scale=self.scale, seed=day_seed, day_var=self.day_var,
                                      wp_name="wp_episode.csv")

    def _settings(self, files, day_seed):
        settings = {
            "rlEnv": 1,
            "sitesFile": files["sites"],
            "waypoints": files["waypoints"],
            "ratesFile": files["rates"],
            "datasetMinutes": self.minutes,
            "rlAntennaInit": 16,
            "isacSenseRadius": 100.0,
            "v2ConnDelayMean": 3.0,
            "v2ActivityProb": 0.7,
            "v2RateCv": 0.3,
            "v2Pd": 0.9,
            "rlSinrThresholdDb": self.sinr_threshold,
            "rlOutageSamplesPerMin": self.outage_samples_per_min,
            "rlOutageEventSec": self.outage_event_sec,
            "RngRun": day_seed + 1,
        }
        settings.update(self.ns3_args)
        return settings

    def reset(self, *, seed=None, options=None):
        options = options or {}
        if "scenario" in options:
            episode = (options["scenario"], int(options["day_seed"]))
        elif not self._started:
            episode = self._first_episode
        else:
            episode = self._sample_train()
        if self._started or episode != self._first_episode:
            files = self._prepare(*episode)
            self.ns3.configure(self._settings(files, episode[1]))
        self._started = True
        self.episode = {"scenario": episode[0], "day_seed": episode[1]}
        # ISAC 오차 난수를 (시나리오, 날)로 정한다. 한 프로세스 안에서 이어지면 같은 날이라도 앞에
        # 몇 에피소드를 돌렸는지(평가 순서, 병렬 배분)에 따라 TCN 입력이 달라진다.
        self.env.reseed([zlib.crc32(episode[0].encode()), episode[1]])
        obs, info = self.env.reset(seed=seed)
        info = dict(info)
        info.update(self.episode)
        return obs, info
