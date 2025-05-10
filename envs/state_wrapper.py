import numpy as np
import torch

import genesis as gs
from envs.vec_env import VecEnv
from utils import *

class StateEnv(VecEnv):
    def _init_buffers(self):
        super()._init_buffers()
        # Add additional buffers here
        # _init_buffers is called once when the environment is created

    def _update_buffers(self):
        super()._update_buffers()
        # update additional buffers here
        # _update_buffers is called every simulation step

    def post_physics_step(self):
        super().post_physics_step()
        # add additional operations after each simulator step here

    def compute_observation(self):
        super().compute_observation()
        # When changing the observation space, make sure to change the num_obs in the config file

    def compute_critic_observation(self):
        super().compute_critic_observation()
        # When changing the critic observation space, make sure to change the num_priv_obs in the config file

    # Add additional methods here, i.e. self._update_desired_contact_states() self._update_feet_pos_local()