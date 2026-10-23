"""ns-3 안테나 제어 RL 환경 (Gymnasium 래퍼).

환경(이음5G 무선, 용량, 전력)은 README_env.md, State와 보상의 정의는 README_AI.md 4절에 있다.
  - ns-3 전파: 안테나당 출력 일정(총 송신 전력이 활성 안테나 수에 비례), 빔포밍 이득 10log10(A)는
    서빙 링크에만 (cttc-nr-demo.cc CoverageModel).
  - 용량 C_max 580 Mbps(ITU-R M.2410), 1인당 수요는 같은 배수 x2.9 (build_tcn_dataset.py).
  - RL State에는 TCN 예측과 함께 TCN 맥락 임베딩 z(차원 = TCN 폭, 지금 64)를 넣는다
    (제안서 State = [y_hat, z, ...]). z는 ReLU 출력이라 가끔 크게 튀므로, TCN 학습 구간
    (0~10, 15~30일)에서 구한 차원별 평균·표준편차로 표준화하고 +-clip에서 자른다(z_stats.json).

ns3-ai가 지원하는 raw 형식(Box)을 감싸서, 에이전트에게는 표준적인 MultiDiscrete
행동 공간과 셀 단위로 정리된 Dict 관측을 제공한다. 래퍼가 흡수하는 것:

  - MultiDiscrete -> Box(uint) 변환. ns3-ai에는 MultiDiscrete가 없어 Box를 쓰는데,
    Box를 그대로 노출하면 SB3 등이 연속 행동 공간으로 오인해 실수값을 내놓는다.
  - 배치 차원 제거 / numpy 스칼라 -> Python int 변환. ns3-ai의 _pack_data가
    len(actions)로 shape를 잡고 protobuf extend()에 그대로 넘기기 때문에,
    (1, 18) 형태나 numpy 스칼라가 들어가면 조용히 틀린 값이 전송된다.
  - 제어 주기(기본 5분, 엔진스펙 3절) 동안 같은 행동을 유지하는 action repeat.
    ns-3 쪽 1 step은 1분이며, 이는 TCN이 학습한 분 단위 관측 해상도를 유지하기 위함이다.
  - 파생 특징 계산. 상수는 build_tcn_dataset.py를 그대로 import 해서 학습 데이터와
    어긋나지 않는다. 단 용량은 안테나 수에 따라 C = C_max * (A / 16) 으로 스케일된다.
  - 셀별 보상 계산. 제어 주기의 모든 분을 누적하고, 18개 셀의 보상 합이 정확히
    캠퍼스 전체 보상이 되도록 나눈다. 과부하 페널티(엔진스펙 8절, 사용률 0.8 초과)를 포함한다.
  - TCN 예측(forecaster)과 이웃 정보로 셀당 RL State를 조립한다.

에피소드는 episode_minutes에 도달하면 바로 끝난다. 이웃에 미치는 영향(엔진스펙 8절 인접 셀
과부하 등)은 학습할 때 train_ppo.py가 이웃 셀 보상을 섞어서(공간 할인) 반영한다.
"""

import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import build_tcn_dataset as btd  # noqa: E402  파생특징 상수를 학습 파이프라인과 공유

ANTENNA_CHOICES = (2, 4, 8, 16)
OBS_PER_CELL = 10
RAW_FIELDS = ("traffic_demand_mbps", "ue_count", "isac_count", "presence_count",
              "mean_sinr_db", "ho_in", "ho_out", "antenna_count",
              "outage_ue", "outage_traffic_mbps")
TCN_FIELDS = ("Traffic_demand_mbps", "Ue_count", "Isac_density", "Hour_sin", "Hour_cos")
RL_FIELDS = ("antenna_level", "Mean_sinr_db", "Prb_utilization", "Neighbor_mean_load")
STATE_FIELDS = (tuple(f"forecast_{h:02d}" for h in range(1, 16))
                + ("forecast_peak", "load", "antenna_level", "hour_sin", "hour_cos",
                   "mean_sinr", "nbr_forecast_peak", "nbr_antenna_level", "nbr_load",
                   "nbr_prb_max"))
# ns-3 extra info의 분 사이 끊김 샘플 합계(캠퍼스 전체) -> info 키
OUTAGE_SAMPLE_KEYS = (("conn", "connected_samples"), ("out", "outage_samples"),
                      ("ev", "outage_events"), ("evl", "outage_events_long"))


def parse_extra_info(info):
    """ns-3 extra info 문자열 "minute=M;spm=S;conn=C;..."을 {키: 값}으로 바꾼다."""
    text = info.get("info", "") if isinstance(info, dict) else ""
    values = {}
    for part in str(text).split(";"):
        key, sep, val = part.partition("=")
        if sep:
            values[key.strip()] = float(val)
    return values


@dataclass
class RewardConfig:
    """보상 파라미터. 기본값은 보고한 에이전트(r10)의 설정이다.

    가중치는 에너지 : QoS : 과부하 = 1:10:5를 합이 1이 되게 나눈 값(0.0625 / 0.625 / 0.3125)이라
    보상이 0~1 범위에 놓인다. 학습할 때는 train_ppo.py --reward-scale 16(정수 비율의 합)을 곱해
    정수 가중치와 같은 학습 신호를 쓴다.

    w_overload는 "받는 셀의 사용률이 U_th=0.8을 넘지 않는다"는 조건을 소프트 제약으로
    옮긴 것이다. 과부하는 사용률이 넘은 그 셀에 귀속하고, 부하를 떠넘긴 이웃은 학습 시
    이웃 보상을 섞는 방식(train_ppo.py --spatial-alpha)으로 그 벌점을 함께 받는다.
    """
    w_energy: float = 0.0625
    w_qos: float = 0.625
    w_overload: float = 0.3125
    u_th: float = 0.8
    static_power_ratio: float = 0.3  # TR 38.864 정적 비율 0.295 (cell_power 참고)


def cell_power(antenna, static_ratio):
    """셀 전력 (안테나 16일 때 1). P = 고정분 + 안테나 비례분.

    3GPP TR 38.864(Rel-18 네트워크 에너지 절감) 5.1절의 공간 영역 적응 모델을 따른다:
    하향 활성 전력은 활성 TxRU 비율에 비례하는 부분과 고정분으로 나뉘고, 고정분 비율은
    기준 구성(FR1 TDD 100 MHz)에서 0.196(기본) / 0.295(정적 전력 1.5배 옵션)이다.
    부하와 무관한 활성 상태(full buffer) 전력이라, 절감률은 활성 상태 절감의 상한이다.
    TR의 슬립 상태까지 넣으면 용량이 안테나 수에 비례하는 스펙(C = C_max x A/16)에서는
    전송 시간이 그만큼 늘어 안테나 조절 절감이 0이 되므로 쓰지 않는다.
    """
    return static_ratio + (1.0 - static_ratio) * np.asarray(antenna) / max(ANTENNA_CHOICES)


class Ns3AntennaEnv(gym.Wrapper):
    """ns3-ai raw 환경을 감싸 셀 단위 안테나 제어 환경으로 노출한다."""

    def __init__(self, env, forecaster, n_cells=18, control_period=5, tcn_window=60,
                 reward_cfg=None, episode_minutes=None,
                 include_z=False, z_stats=None, seed=None):
        super().__init__(env)
        self.n_cells = n_cells
        self.control_period = control_period
        self.tcn_window = tcn_window
        self.reward_cfg = reward_cfg or RewardConfig()
        self.episode_minutes = episode_minutes
        self.forecaster = forecaster
        self.include_z = include_z
        if include_z and z_stats is None:
            raise ValueError("include_z=True에는 z_stats(mean, std, clip)가 필요합니다")
        self._z_mean = np.asarray(z_stats["mean"], np.float32) if include_z else None
        self._z_std = np.asarray(z_stats["std"], np.float32) if include_z else None
        self._z_clip = float(z_stats.get("clip", 5.0)) if include_z else None
        self._rng = np.random.default_rng(seed)
        self._nbrs = self._neighbor_index()
        self._hist = deque(maxlen=tcn_window)

        if tuple(forecaster.feature_names) != TCN_FIELDS:
            raise ValueError(f"TCN 입력 채널 {forecaster.feature_names}가 "
                             f"환경의 tcn_window 채널 {TCN_FIELDS}와 다릅니다")

        self.action_space = spaces.MultiDiscrete([len(ANTENNA_CHOICES)] * n_cells)
        space = {
            "tcn_window": spaces.Box(-1e6, 1e6, (n_cells, len(TCN_FIELDS), tcn_window), np.float32),
            "rl_state": spaces.Box(-1e6, 1e6, (n_cells, len(RL_FIELDS)), np.float32),
            "raw": spaces.Box(-1e6, 1e6, (n_cells, OBS_PER_CELL), np.float32),
            "minute": spaces.Box(0, 1e4, (1,), np.float32),
        }
        dim = len(STATE_FIELDS) + (forecaster.z_dim if include_z else 0)
        space["forecast"] = spaces.Box(-1e6, 1e6, (n_cells, forecaster.horizon + 1), np.float32)
        space["state"] = spaces.Box(-1e6, 1e6, (n_cells, dim), np.float32)
        self.observation_space = spaces.Dict(space)

    def _neighbor_index(self):
        """build_tcn_dataset의 최근접 3셀 정의를 0-based 인덱스 배열로 변환."""
        meta = btd.cell_meta()
        nbrs = btd.neighbor_lists(meta, btd.N_NEIGHBORS)
        return [np.array([j - 1 for j in nbrs[c]], dtype=np.int64)
                for c in sorted(meta.keys())]

    def _nbr_mean(self, values):
        return np.array([values[idx].mean() if idx.size else 0.0 for idx in self._nbrs],
                        dtype=np.float32)

    def _nbr_max(self, values):
        return np.array([values[idx].max() if idx.size else 0.0 for idx in self._nbrs],
                        dtype=np.float32)

    @staticmethod
    def _served_utilization(raw):
        """정상(끊기지 않은) UE 수요 기준 사용률 U = min(1, (D - D_out) / C)."""
        capacity = btd.CELL_CMAX_MBPS * (raw[:, 7] / btd.BASELINE_ANTENNAS)
        served = np.maximum(raw[:, 0] - raw[:, 9], 0.0)
        return np.minimum(1.0, served / np.maximum(capacity, 1e-9)), served, capacity

    def _to_ns3_action(self, action):
        """MultiDiscrete 행동을 ns3-ai가 요구하는 평평한 Python int 리스트로 변환."""
        a = np.asarray(action).reshape(-1).astype(np.int64)
        if a.size != self.n_cells:
            raise ValueError(f"행동 길이가 {a.size}, {self.n_cells}이어야 함")
        return np.clip(a, 0, len(ANTENNA_CHOICES) - 1).tolist()

    def _decode(self, raw_obs):
        flat = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
        expected = self.n_cells * OBS_PER_CELL + 1
        if flat.size != expected:
            raise ValueError(f"관측 길이가 {flat.size}, {expected}이어야 함 "
                             "(ns-3를 다시 빌드했는지 확인)")
        cells = flat[:self.n_cells * OBS_PER_CELL].reshape(self.n_cells, OBS_PER_CELL)
        minute = float(flat[-1])

        traffic, ue_count, isac_count = cells[:, 0], cells[:, 1], cells[:, 2]
        sinr, antenna = cells[:, 4], cells[:, 7]

        capacity = btd.CELL_CMAX_MBPS * (antenna / btd.BASELINE_ANTENNAS)
        prb = np.minimum(1.0, traffic / np.maximum(capacity, 1e-9))
        neighbor_load = self._nbr_mean(prb)

        isac_density = np.maximum(
            0.0, isac_count * (1.0 + self._rng.normal(0.0, btd.ISAC_SIGMA, self.n_cells)))
        hour = 9.0 + minute / 60.0
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)

        tcn_frame = np.stack([
            traffic, ue_count, isac_density.astype(np.float32),
            np.full(self.n_cells, hour_sin, np.float32),
            np.full(self.n_cells, hour_cos, np.float32),
        ], axis=1).astype(np.float32)
        self._hist.append(tcn_frame)

        level = np.searchsorted(np.array(ANTENNA_CHOICES), antenna).astype(np.float32)
        rl_state = np.stack([level, sinr, prb.astype(np.float32), neighbor_load], axis=1)

        return {
            "tcn_window": self._window(),
            "rl_state": rl_state.astype(np.float32),
            "raw": cells.astype(np.float32),
            "minute": np.array([minute], np.float32),
        }

    def _window(self):
        """최근 tcn_window분을 (셀, 특징, 시간)으로 쌓는다. 초반에는 첫 프레임으로 좌측 패딩."""
        frames = list(self._hist)
        if not frames:
            return np.zeros((self.n_cells, len(TCN_FIELDS), self.tcn_window), np.float32)
        pad = [frames[0]] * (self.tcn_window - len(frames))
        return np.stack(pad + frames, axis=2).astype(np.float32)

    def _add_state(self, obs):
        """TCN 예측과 이웃 정보로 셀당 RL State를 만든다. 부하 단위는 C_max 대비 비율."""
        central, peak, z = self.forecaster.predict(obs["tcn_window"])
        central = np.maximum(central, 0.0)
        peak = np.maximum(peak, 0.0)
        cmax = btd.CELL_CMAX_MBPS
        raw = obs["raw"]
        load = raw[:, 0] / cmax
        level = obs["rl_state"][:, 0] / (len(ANTENNA_CHOICES) - 1)
        hour_sin = obs["tcn_window"][:, 3, -1]
        hour_cos = obs["tcn_window"][:, 4, -1]
        util, _, _ = self._served_utilization(raw)
        parts = [
            central / cmax,
            (peak / cmax)[:, None],
            load[:, None],
            level[:, None],
            hour_sin[:, None],
            hour_cos[:, None],
            (raw[:, 4] / 20.0)[:, None],
            self._nbr_mean(peak / cmax)[:, None],
            self._nbr_mean(level)[:, None],
            self._nbr_mean(load)[:, None],
            # 과부하 페널티가 이웃 사용률로 결정되므로 정책이 그것을 직접 볼 수 있게 한다.
            self._nbr_max(util)[:, None],
        ]
        if self.include_z:
            parts.append(np.clip((z - self._z_mean) / self._z_std, -self._z_clip, self._z_clip))
        obs["forecast"] = np.concatenate([central, peak[:, None]], axis=1).astype(np.float32)
        obs["state"] = np.concatenate(parts, axis=1).astype(np.float32)
        return obs

    def reseed(self, seed):
        """ISAC 측정 오차 난수를 다시 정한다. 에피소드마다 날로 정하면 결과가 실행 순서와 무관해진다."""
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        self._hist.clear()
        raw_obs, info = self.env.reset(seed=seed, options=options)
        return self._add_state(self._decode(raw_obs)), info

    def step(self, action):
        """행동을 control_period분 동안 유지하며 내부 환경을 그만큼 전진시킨다.

        반환하는 obs는 마지막 분의 스냅샷이고, 보상과 info의 지표는 제어 주기의 매 분을
        누적한 값이다. 셀 c의 보상 (모든 항은 0~1로 정규화, N = 셀 수):

            r_c = -( w_E * E_c  +  w_Q * S_c / D_bar  +  w_O * O_c ) / N

          E_c   : 셀 c의 평균 전력 (안테나 16 = 1)
          S_c   : 셀 c의 부족분 합 = sum_분 [ max(0, 정상 UE 수요 - 용량) + 끊긴 UE 수요 ]
          D_bar : 캠퍼스 전체 요구 트래픽 합 / N
          O_c   : 셀 c의 과부하 = max(0, U_c - u_th) / (1 - u_th) 의 5분 평균

        sum_c r_c = -(w_E * E + w_Q * Q + w_O * O) 로 캠퍼스 전체 보상과 정확히 같다.
        반환하는 스칼라 보상은 셀 평균(= 전체 보상 / N)이고, 셀별 값은 info["cell_reward"].

        분 사이 끊김 샘플(ns-3 --rlOutageSamplesPerMin, 보상에는 쓰지 않음)은 제어 주기 동안
        캠퍼스 합계로 누적한다: connected_samples(접속 UE 샘플), outage_samples(그중 끊김),
        outage_events(끊김 시작 횟수), outage_events_long(--rlOutageEventSec 이상 이어진 끊김,
        끊김 길이 = 샘플 수 x 간격이라 간격이 기준 이상이면 측정 불가로 nan), outage_sample_sec
        (샘플 간격, 초).
        """
        ns3_action = self._to_ns3_action(action)
        cfg = self.reward_cfg
        obs, terminated, truncated, info = None, False, False, {}
        n = self.n_cells
        ho_in, ho_out = np.zeros(n), np.zeros(n)
        demand, shortage = np.zeros(n), np.zeros(n)
        energy_c, overload_c = np.zeros(n), np.zeros(n)
        ue, outage_ue = np.zeros(n), np.zeros(n)
        over_count, minutes = 0, 0
        samples = dict.fromkeys(dict(OUTAGE_SAMPLE_KEYS).values(), 0.0)
        samples_per_min = 0.0
        long_measurable = True
        for _ in range(self.control_period):
            raw_obs, _, terminated, truncated, info = self.env.step(ns3_action)
            extra = parse_extra_info(info)
            for key, name in OUTAGE_SAMPLE_KEYS:
                samples[name] += extra.get(key, 0.0)
            samples_per_min = extra.get("spm", 0.0)
            # ns-3는 샘플 간격이 긴 끊김 기준 이상이면 evl=-1(측정 불가)을 보낸다
            long_measurable = long_measurable and extra.get("evl", 0.0) >= 0
            obs = self._decode(raw_obs)
            raw = obs["raw"].astype(np.float64)
            util, served, capacity = self._served_utilization(raw)
            over = np.maximum(0.0, util - cfg.u_th) / (1.0 - cfg.u_th)
            shortage += np.maximum(0.0, served - capacity) + raw[:, 9]
            demand += raw[:, 0]
            ue += raw[:, 1]
            outage_ue += raw[:, 8]
            ho_in += raw[:, 5]
            ho_out += raw[:, 6]
            energy_c += cell_power(raw[:, 7], cfg.static_power_ratio)
            overload_c += over
            over_count += int((util > cfg.u_th).sum())
            minutes += 1
            # 하루의 마지막 분을 받으면 바로 끝낸다. ns-3는 다음 reset의 종료 요청으로 정리된다.
            if self.episode_minutes is not None and obs["minute"][0] >= self.episode_minutes:
                terminated = True
            if terminated or truncated:
                break
        if not long_measurable:
            samples["outage_events_long"] = float("nan")
        m = max(minutes, 1)
        energy_c /= m
        overload_c /= m
        d_bar = demand.sum() / n
        qos_c = shortage / d_bar if d_bar > 0 else np.zeros(n)
        term_e = cfg.w_energy * energy_c / n
        term_q = cfg.w_qos * qos_c / n
        term_o = cfg.w_overload * overload_c / n
        cell_reward = -(term_e + term_q + term_o)
        info = dict(info)
        info.update({
            "cell_reward": cell_reward.astype(np.float32),
            "energy": float(energy_c.mean()),
            "shortage_ratio": float(shortage.sum() / demand.sum()) if demand.sum() > 0 else 0.0,
            "overload": float(overload_c.mean()),
            "overload_frac": over_count / float(n * m),
            "term_energy": float(term_e.sum()),
            "term_qos": float(term_q.sum()),
            "term_overload": float(term_o.sum()),
            "shortage_mbps": shortage,
            "demand_mbps": demand,
            "drop_rate": float(outage_ue.sum() / ue.sum()) if ue.sum() > 0 else 0.0,
            "ue_total": float(ue.sum()),
            "outage_ue_total": float(outage_ue.sum()),
            "ho_in_sum": ho_in,
            "ho_out_sum": ho_out,
            "minutes": minutes,
            **samples,
            # 이전 ns-3 바이너리는 샘플 합계를 보내지 않는다 (간격 nan)
            "outage_sample_sec": 60.0 / samples_per_min if samples_per_min > 0 else float("nan"),
        })
        return self._add_state(obs), float(cell_reward.mean()), terminated, truncated, info
