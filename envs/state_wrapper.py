import numpy as np
import torch

import genesis as gs
from envs.vec_env import VecEnv

class StateEnv(VecEnv):
    def __init__(self, num_envs, env_cfg, show_viewer, eval, debug, n_rendered_envs=1, device='cuda'):
        super().__init__(num_envs, env_cfg, show_viewer, eval, debug, n_rendered_envs, device)
        # Initialize the environment

    def _init_buffers(self):
        super()._init_buffers()
        # Add additional buffers here

    def _update_buffers(self):
        super()._update_buffers()
        # update additional buffers here

    def post_physics_step(self):
        super().post_physics_step()
        # add additional operations after each simulator step here

    def compute_observation(self):
        super().compute_observation()

    def compute_critic_observation(self):
        super().compute_critic_observation()