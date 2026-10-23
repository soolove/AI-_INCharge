"""학습된 dual-head TCN을 불러와 RL State용 예측을 만든다.

train_dual.py로 학습한 체크포인트를 그대로 쓴다. 체크포인트 안의 정규화 통계를 적용해
입력을 표준화하고, 출력은 Mbps로 되돌린다. TCN은 고정(freeze)이며 RL 학습 중에 갱신되지
않는다. 모델 폭(z 차원)은 체크포인트의 구조 이름에서 읽는다.

기본 TCN = 폭 64, tcn-dataset-v3-5g-31d(5G 단위, 접속 지연 수정)로 학습한 모델.
"""

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
_TCN_SRC = _REPO / "DGU_SC_Project"
for _p in (_TCN_SRC, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tcn_dual import DualHeadTCN, width_from_arch  # noqa: E402

DEFAULT_CKPT = _TCN_SRC / "checkpoints/v3_5g_31d_w64_dual_proposal/seed_0_finetune/best.pt"


class TcnForecaster:
    """셀별 과거 60분 창 -> 향후 15분 트래픽(central), 15분 최대 부하(peak), 맥락 z."""

    def __init__(self, ckpt=DEFAULT_CKPT, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        ckpt = torch.load(ckpt, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        if cfg.get("loss") != "dual_pinball":
            raise ValueError("train_dual.py로 학습한 dual-head 체크포인트가 필요합니다")
        self.feature_names = list(cfg["feature_names"])
        # 폭(채널 수)은 체크포인트의 구조 이름에 들어 있다. z 차원 = 폭.
        self.model = DualHeadTCN(cfg["input_channels"],
                                 channels=width_from_arch(cfg.get("architecture"))
                                 ).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.stats = {k: v.to(self.device) for k, v in ckpt["normalization"].items()}
        self.z_dim = self.model.central_head.in_features
        self.horizon = self.model.central_head.out_features

    @torch.inference_mode()
    def predict(self, window):
        """window (n_cells, C, 60) -> central (n,15) Mbps, peak (n,) Mbps, z (n, z_dim)."""
        x = torch.tensor(np.asarray(window, dtype=np.float32), device=self.device)
        s = self.stats
        z = self.model.tcn((x - s["x_mean"]) / s["x_std"])[:, :, -1]
        central = self.model.central_head(z) * s["y_std"] + s["y_mean"]
        peak = self.model.peak_head(z).squeeze(-1) * s["m_std"] + s["m_mean"]
        return central.cpu().numpy(), peak.cpu().numpy(), z.cpu().numpy()
