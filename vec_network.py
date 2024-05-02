from collections.abc import Callable, Sequence
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F


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
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(inplace=True),
        )
        self.output_net = nn.Sequential(
            layer_init(nn.Linear(256, self.action_num * self.num_quantiles)),
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



class Actor(nn.Module):
    """Actor network for discrete action space"""
    def __init__(self, state_size, action_size, hidden_size=32):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits
    def get_action(self, state):
        action_logits = self.forward(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        return action.detach().cpu()
    def evaluate(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist

# class Critic(nn.Module):
#     """Critic network for discrete action space"""
#     def __init__(self, state_size, action_size, hidden_size=32):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(state_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size + action_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, 1)
#     def forward(self, state, action):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(torch.cat([x, action], dim=1)))
#         return self.fc3(x)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Value(nn.Module):
    """Value (Value) Model."""

    def __init__(self, state_size, hidden_size=32):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)