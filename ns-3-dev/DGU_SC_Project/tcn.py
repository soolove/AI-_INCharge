"""Causal TCN building blocks (CausalConv1d, TemporalBlock) used by tcn_dual.py."""

from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """Map [B, C_in, T] to [B, C_out, T] without reading future steps."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

    def forward(self, x):
        # Left-only padding preserves T and prevents future information leakage.
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TemporalBlock(nn.Module):
    """Two causal convolutions with a residual path; time length is unchanged."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.residual(x))

