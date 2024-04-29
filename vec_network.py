from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QRDQN(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_shape: Sequence[int] | int,
        num_quantiles: int = 200,
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.action_num = int(np.prod(action_shape))
        self.num_quantiles = num_quantiles
        self.feature_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(inplace=True),
        )
        self.output_net = nn.Sequential(
            layer_init(nn.Linear(128, self.action_num * self.num_quantiles)),
        )

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        features = self.feature_net(obs)
        output = self.output_net(features).view(-1, self.action_num, self.num_quantiles)
        return output, state