#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .ppo import PPO
from .tdo import TDO

__all__ = ["PPO", "TDO"]
