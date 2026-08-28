"""
T-Bidirectional Vision (TBV) - Direction Modulation
Direction signal d in {+1, -1} is mapped to channel-wise scale (gamma) and bias (beta).
"""

import torch
import torch.nn as nn
from typing import Union


class DirectionModulation(nn.Module):
    """
    Direction Modulation module for TBlock.
    Maps direction index (+1 -> 0, -1 -> 1) through an embedding layer
    to produce scale gamma and bias beta vectors for feature channels.
    """
    def __init__(self, dim: int, emb_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim

        # Index 0: +1 (Image -> Latent)
        # Index 1: -1 (Latent -> Image)
        self.emb = nn.Embedding(2, emb_dim)
        self.fc = nn.Linear(emb_dim, dim * 2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.emb.weight, std=0.02)
        # Initialize to small weight & zero bias for identity modulation
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, direction: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, D, H, W)
            direction: (B,) long tensor with values 0 (+1) or 1 (-1), or scalar integer.
        Returns:
            Modulated tensor (B, D, H, W)
        """
        if isinstance(direction, int):
            dir_tensor = torch.tensor([direction], dtype=torch.long, device=x.device).expand(x.shape[0])
        elif isinstance(direction, torch.Tensor):
            if direction.dim() == 0:
                dir_tensor = direction.long().expand(x.shape[0])
            elif direction.dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
                # map +1 -> 0, -1 -> 1 if given as signed values
                if (direction == 1).any() and (direction == -1).any():
                    dir_tensor = torch.where(direction > 0, 0, 1).long()
                else:
                    dir_tensor = direction.long()
            else:
                dir_tensor = direction.long()
        else:
            dir_tensor = torch.tensor([0 if direction >= 0 else 1], dtype=torch.long, device=x.device).expand(x.shape[0])

        e = self.emb(dir_tensor)  # (B, emb_dim)
        gb = self.fc(e)           # (B, 2 * dim)
        gamma, beta = gb.chunk(2, dim=-1)  # (B, dim), (B, dim)

        # Add +1 to gamma so default scale is 1.0 (identity baseline)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) + 1.0  # (B, dim, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)          # (B, dim, 1, 1)

        return gamma * x + beta
