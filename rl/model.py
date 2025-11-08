from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover - allow import failure in runtime container
    torch = None
    nn = object  # type: ignore
    F = None  # type: ignore


class DQN(nn.Module):  # type: ignore[misc]
    """
    Tiny convolutional Q-network for grid-based observations with optional scalar features.
    Inputs:
      - grid_tensor: shape (B, C, H, W), C ~ 3-5 feature planes
      - scalar_tensor: shape (B, S) or None, for boosts, turn count, etc.
    Outputs:
      - q_values: shape (B, A) where A is number of discrete actions (4 or 8 incl. boost)
    """

    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        num_scalar_features: int = 0,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if torch is None:
            raise RuntimeError("PyTorch is required to construct DQN.")

        # Very small conv stack for 18x20 grid
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Compute flattened size assuming fixed grid (18x20)
        example = torch.zeros(1, in_channels, 18, 20)
        with torch.no_grad():
            conv_out = self.conv(example)
        conv_flattened = conv_out.view(1, -1).shape[1]

        self.num_scalar_features = num_scalar_features
        head_in = conv_flattened + (num_scalar_features if num_scalar_features > 0 else 0)

        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, grid_tensor, scalar_tensor: Optional["torch.Tensor"] = None):  # type: ignore[name-defined]
        x = self.conv(grid_tensor)  # (B, C, H, W)
        x = x.view(x.size(0), -1)
        if self.num_scalar_features > 0 and scalar_tensor is not None:
            x = torch.cat([x, scalar_tensor], dim=1)
        return self.head(x)


