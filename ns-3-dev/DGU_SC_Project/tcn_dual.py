"""Dual-head causal TCN.

TCN backbone of 4 temporal blocks (tcn.py) with two output heads:
  - central head: [B, 15] per-minute traffic, trained toward the p50 quantile
  - peak head:    [B]     the 15-minute maximum load, trained toward a high quantile (p90)

Both heads read z, the backbone output at the last time step (the context embedding the RL
State uses).
"""

import re

import torch
from torch import nn

from tcn import TemporalBlock


def width_from_arch(arch):
    """Channel width in an architecture string ("tcn-dual-64x4-..."); 32 if absent."""
    match = re.match(r"tcn-dual-(\d+)x4-", str(arch))
    return int(match.group(1)) if match else 32


class DualHeadTCN(nn.Module):
    """Predict a central [B,15] series and a single [B] peak from [B,C,60].

    channels is the width of the 4 temporal blocks and of z (32 = the original AI-team model,
    64 = the TCN used in this project).
    Depth stays at 4 blocks: dilations 1/2/4/8 already give a 61-minute receptive field
    for the 60-minute input.
    """

    def __init__(self, input_channels, channels=32):
        super().__init__()
        if not isinstance(input_channels, int) or input_channels < 1:
            raise ValueError("input_channels must be a positive integer")
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        self.input_channels = input_channels
        widths = (channels,) * 4
        dilations = (1, 2, 4, 8)
        kernel_size = 3
        blocks = []
        in_channels = input_channels
        for out_channels, dilation in zip(widths, dilations):
            blocks.append(TemporalBlock(
                in_channels, out_channels, kernel_size, dilation, dropout=0.1
            ))
            in_channels = out_channels
        self.tcn = nn.Sequential(*blocks)
        self.central_head = nn.Linear(channels, 15)  # per-minute traffic
        self.peak_head = nn.Linear(channels, 1)       # 15-minute maximum load
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.input_channels, 60):
            raise ValueError(
                f"Expected [B, {self.input_channels}, 60], got {tuple(x.shape)}"
            )
        features = self.tcn(x)[:, :, -1]        # [B, channels] last timestep
        central = self.central_head(features)   # [B, 15]
        peak = self.peak_head(features).squeeze(-1)  # [B]
        return central, peak


def pinball_loss(prediction, target, tau):
    """Quantile (pinball) loss for target quantile tau in (0, 1).

    u = target - prediction; loss = mean(max(tau*u, (tau-1)*u)).
    tau=0.5 gives (half the) mean absolute error; a higher tau penalises
    underprediction more, so the prediction rises toward the tau-quantile.
    """
    u = target - prediction
    return torch.maximum(tau * u, (tau - 1.0) * u).mean()