import numpy as np
import torch
import torch.nn as nn

import genesis as gs
from rsl_rl.env import VecEnv

class TimeWrapper:
    def __init__(self, env: VecEnv, period_length: int, reset_each_period: bool, observe_time: bool):
        self.env = env
        self.period_length = period_length
        self.reset_each_period = reset_each_period
        self.observe_time = observe_time

        self.device = env.device
        self.num_envs = env.num_envs
        self.num_states = env.num_states
        if self.observe_time:
            self.num_obs = env.num_obs + 6
            self.num_privileged_obs = env.num_privileged_obs + 6

        self.time_buf = torch.zeros(
            (self.num_envs), device=self.device, dtype=gs.tc_int
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        obs, info = self.env.get_observations()
        time_obs = self._get_time_obs()
        if self.observe_time:
            obs = torch.cat(
                [
                    obs,
                    time_obs,
                ],
                dim=-1
            )
            info["critic_obs"] = torch.cat(
                [
                    info["observations"]["critic"],
                    time_obs,
                ],
                dim=-1
            )
        return obs, info

    def get_privileged_observations(self):
        privileged_obs = self.env.get_privileged_observations()
        if self.observe_time:
            time_obs = self._get_time_obs()
            privileged_obs = torch.cat(
                [
                    privileged_obs,
                    time_obs,
                ],
                dim=-1
            )
        return privileged_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.time_buf = torch.remainder(self.time_buf + 1, self.period_length)
        if self.reset_each_period:
            done = torch.logical_or(done, self.time_buf == 0)

        if self.observe_time:
            time_obs = self._get_time_obs()
            obs = torch.cat(
                [
                    obs,
                    time_obs,
                ],
                dim=-1
            )
            info["critic_obs"] = torch.cat(
                [
                    info["observations"]["critic"],
                    time_obs,
                ],
                dim=-1
            )

        return obs, reward, done, info

    def set_state(self, states, envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self.device)
        self.env.set_state(states, self.time_buf[envs_idx], envs_idx)

    def set_time(self, times, envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self.device)
        self.time_buf[envs_idx] = times
        self.env.episode_length_buf[envs_idx] = times

    def get_state(self):
        state, _ = self.env.get_state()
        return state, self.time_buf

    def get_time(self, env_idx):
        return self.time_buf[env_idx]

    def _get_time_obs(self, times=None):
        if times is None:
            times = self.time_buf
        phase = times.float().unsqueeze(1) / self.period_length * 2 * np.pi
        time_obs = torch.cat([
            torch.sin(phase),
            torch.cos(phase),
            torch.sin(phase / 2),
            torch.cos(phase / 2),
            torch.sin(phase / 4),
            torch.cos(phase / 4),
        ], dim=-1)
        return time_obs