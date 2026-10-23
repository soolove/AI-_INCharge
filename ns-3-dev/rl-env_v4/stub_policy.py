"""비교 기준(baseline) 정책: 상시 최대(constant_max)와 반응형(threshold).

제안서 4절이 요구하는 두 비교군(절전 미적용 / 현재 부하 기반 반응형 제어)이다. 에이전트와 같은
시그니처 policy(obs) -> (n_cells,) 정수 배열(안테나 인덱스 0..3)을 지킨다.
"""

import numpy as np

N_LEVELS = 4  # {2, 4, 8, 16}


def constant_max(obs, n_cells=18):
    """항상 최대 안테나. 제안서의 비교군 (1) 절전 미적용."""
    return np.full(n_cells, N_LEVELS - 1, dtype=np.int64)


def threshold(obs, n_cells=18, up=0.03, down=0.02):
    """현재 PRB 사용률만 보고 단계를 올리고 내리는 반응형 제어.

    제안서의 비교군 (2) 현재 부하 기반 반응형 제어에 해당한다. 미래를 보지 않으므로
    부하가 오른 뒤에야 반응하며, 이것이 TCN 기반 선행 제어와의 대조군이 된다.
    기본 임계값은 tune_reactive.py가 val 날에서 고른 값이다(runs/reactive_tuned_v2_r10.json).
    """
    prb = obs["rl_state"][:, 2]
    level = obs["rl_state"][:, 0].astype(np.int64)
    level = np.where(prb > up, level + 1, level)
    level = np.where(prb < down, level - 1, level)
    return np.clip(level, 0, N_LEVELS - 1)
