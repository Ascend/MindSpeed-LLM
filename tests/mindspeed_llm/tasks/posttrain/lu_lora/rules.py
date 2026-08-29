# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn


class OjasRule(nn.Module):
    """
    OjasRule class implementation.
    """

    # Sequence dimension
    _sequence_axis: int

    def __init__(self, sequence_axis: int = 0) -> None:
        """
        Initialize an instance of OjasRule.

        Args:
            sequence_axis (int): axis of a sequence in the input tensor.
        """
        super().__init__()
        self._sequence_axis = sequence_axis

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Forward of the OjasRule.

        Args:
            x (torch.Tensor): Input for the Linear layer with weights W.
            y (torch.Tensor): Result of the forward the Linear layer: y=xW^t.
            weight (torch.Tensor): Weights of the Linear layer.

        Returns:
            torch.Tensor: weights delta
        """
        # delta = ny(x - yw) = n(yx - y^2w)
        if self._sequence_axis == 0:
            x = torch.permute(x, (1, 0, 2))
            y = torch.permute(y, (1, 0, 2))

        yx = y.mean(self._sequence_axis).transpose(0, 1) @ x.mean(self._sequence_axis)
        yw = (y**2).mean(1).mean(0).unsqueeze(1) * weight
        delta = yx - yw
        return F.normalize(delta, dim=0)
