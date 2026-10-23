"""셀 공유 Actor-Critic 정책 (셀별 가치).

18개 셀에 같은 가중치를 적용한다. 셀들은 하드웨어·안테나 선택지·전파 모델이 같아
셀마다 다른 가중치를 둘 이유가 없고, 파라미터가 줄어 표본 효율이 좋아진다. 셀 수가
다른 캠퍼스에도 그대로 쓸 수 있다.

    state (B, N, D)
      -> encoder (셀별, 공유)          h_c  (B, N, H)
      -> 전체 셀 평균                   g    (B, H)       캠퍼스 전체 상황
      -> actor([h_c, g])  (셀별, 공유)  logits (B, N, 4)  셀별 안테나 단계 분포
      -> critic([h_c, g]) (셀별, 공유)  V      (B, N)     셀별 가치

보상이 셀별로 나뉘므로 가치도 셀별로 낸다. 로그확률·엔트로피도 셀별로 돌려주어, PPO가
셀마다 따로 확률 비율을 계산하게 한다. 18개 셀 확률의 곱으로 비율을 계산하면 셀 하나의
작은 변화도 곱해져 clip에 자주 걸린다.
"""

import torch
from torch import nn
from torch.distributions import Categorical


def _layer(i, o, gain=2 ** 0.5):
    layer = nn.Linear(i, o)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class SharedCellActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions=4, hidden=128, init_max_bias=1.5):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.encoder = nn.Sequential(_layer(state_dim, hidden), nn.Tanh(),
                                     _layer(hidden, hidden), nn.Tanh())
        self.actor = nn.Sequential(_layer(2 * hidden, hidden), nn.Tanh(),
                                   _layer(hidden, n_actions, gain=0.01))
        self.critic = nn.Sequential(_layer(2 * hidden, hidden), nn.Tanh(),
                                    _layer(hidden, 1, gain=1.0))
        # 초기 정책을 "대체로 최대 안테나" 쪽에 둔다. 균등 무작위로 시작하면 처음부터 커버리지
        # 공백이 많아 보상이 매우 나쁜 영역만 탐색하게 된다. 상시 최대(안전한 기준선)에서
        # 출발해 줄여도 되는 셀을 찾아가게 하려는 것이다. bias 1.5 -> 최대 단계 확률 약 0.6.
        with torch.no_grad():
            self.actor[-1].bias[-1] = init_max_bias

    def forward(self, state):
        h = self.encoder(state)
        g = h.mean(dim=1, keepdim=True).expand_as(h)
        hg = torch.cat([h, g], dim=-1)
        return self.actor(hg), self.critic(hg).squeeze(-1)

    def act(self, state, deterministic=False):
        """state (B, N, D) -> action (B, N), 셀별 log_prob (B, N), 셀별 value (B, N)."""
        logits, value = self(state)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, state, action):
        """PPO 갱신용: 셀별 log_prob (B, N), 셀별 entropy (B, N), 셀별 value (B, N)."""
        logits, value = self(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value
