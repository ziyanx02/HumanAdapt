import numpy as np
import torch
import torch.nn as nn

import genesis as gs
from envs.state_wrapper import StateEnv

class RewardEnv(StateEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _reward_alive(self):
        """
        Reward for staying alive (not terminating).
        - Returns 1 if the robot is alive (terminate_buf is False), otherwise 0.
        - Encourages the robot to avoid conditions that lead to termination (e.g., falling).
        """
        return 1 - self.terminate_buf.float()

    # Add reward functions here
    # For reward function introduced as REWARD_FUNC in the config file,
    # the function name should be _reward_REWARD_FUNC
    # def _reward_REWARD_FUNC(self):
    #     return torch.zeros(self.num_envs, device=self.device)