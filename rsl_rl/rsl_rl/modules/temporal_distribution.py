#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class TemporalDistribution(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_state,
        period_length,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        learning_rate=1e-3,
        **kwargs,
    ):
        super().__init__()
        self.num_state = num_state
        self.period_length = period_length
        self.learning_rate = learning_rate
        activation = get_activation(activation)
        self.mean_params = nn.Parameter(torch.zeros((period_length, num_state)))
        self.std_params = nn.Parameter(torch.ones((period_length, num_state)))

    def init_params(self, env):
        self.mean_params.data = env.state_mean
        self.std_params.data = env.state_std

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def forward(self):
        raise NotImplementedError

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, times):
        self.std_params.data.clamp_(min=0.01)
        self.distribution = Normal(self.mean_params[times], self.std_params[times])

    def sample(self, times):
        self.update_distribution(times)
        return self.distribution.sample()

    def get_states_log_prob(self, states, times):
        times = times.reshape(-1)
        self.update_distribution(times)
        return self.distribution.log_prob(states).sum(dim=-1)

    def sample_inference(self, times):
        state_mean = self.mean_params[times]
        return state_mean

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
