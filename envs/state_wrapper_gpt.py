
import numpy as np
import torch

import genesis as gs
from envs.vec_env import VecEnv
from utils import *

class StateEnv(VecEnv):
    def _init_buffers(self):
        super()._init_buffers()

        # Ensure feet_link_indices and commands are initialized
        self.feet_link_indices = self._find_link_indices(self.env_cfg['feet_link_names'], accurate=True)

        # Initialize commands
        self.commands = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.commands_scale = torch.tensor(
            [
                self.obs_scales['lin_vel'],
                self.obs_scales['lin_vel'],
                self.obs_scales['ang_vel'],
            ],
            device=self.device,
            dtype=gs.tc_float,
        )

        # Initialize gait-related buffers
        self.foot_positions = torch.ones(
            self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float,
        )
        self.foot_quaternions = torch.ones(
            self.num_envs, len(self.feet_link_indices), 4, device=self.device, dtype=gs.tc_float,
        )
        self.foot_velocities = torch.ones(
            self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float,
        )

        self.gait_indices = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.foot_indices = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )

        self.gait_frequency = torch.ones(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.gait_duration = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.gait_offset = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.gait_feet_height = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.gait_feet_stationary_pos = torch.zeros(
            (self.num_envs, len(self.feet_link_indices), 2), device=self.device, dtype=gs.tc_float,
        )
        self.gait_body_height = torch.zeros(
            self.num_envs, device=self.device, dtype=gs.tc_float,
        )

        self.clock_inputs = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.doubletime_clock_inputs = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.halftime_clock_inputs = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )

        self.desired_contact_states = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.desired_feet_pos_local = torch.zeros(
            (self.num_envs, len(self.feet_link_indices), 3), device=self.device, dtype=gs.tc_float,
        )
        self.feet_pos_local = torch.zeros(
            (self.num_envs, len(self.feet_link_indices), 3), device=self.device, dtype=gs.tc_float,
        )

        self._load_gait(self.env_cfg['gait'])

    def _find_link_indices(self, names, accurate=False):
        link_indices = list()
        availible = [True for i in range(len(self.robot.links))]
        for name in names:
            for i, link in enumerate(self.robot.links):
                if availible[i] and (accurate == False and name in link.name or name == link.name):
                    availible[i] = False
                    link_indices.append(link.idx - self.robot.link_start)
        return link_indices

    def _load_gait(self, gait_cfg):
        self.gait_cfg = gait_cfg
        self.gait_body_height[:] = gait_cfg['base_height_target']
        for i in range(len(self.feet_link_indices)):
            self.gait_frequency[:, i] = gait_cfg['frequency'][i]
            self.gait_duration[:, i] = gait_cfg['duration'][i]
            self.gait_offset[:, i] = gait_cfg['offset'][i]
            self.gait_feet_height[:, i] = gait_cfg['feet_height_target'][i]
            self.gait_feet_stationary_pos[:, i, 0] = gait_cfg['stationary_position'][i][0]
            self.gait_feet_stationary_pos[:, i, 1] = gait_cfg['stationary_position'][i][1]

    def _update_desired_contact_states(self):
        num_feet = len(self.feet_link_indices)

        self.gait_indices = torch.remainder(self.gait_indices + self.dt * self.gait_frequency, 1.0)
        self.foot_indices = torch.remainder(self.gait_indices + self.gait_offset, 1.0)

        for i in range(num_feet):
            idxs = self.foot_indices[:, i]
            duration = self.gait_duration[:, i]
            stance_idxs = torch.remainder(idxs, 1) < duration
            swing_idxs = torch.remainder(idxs, 1) > duration

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / duration[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - duration[swing_idxs]) * (
                        0.5 / (1 - duration[swing_idxs]))

        self.clock_inputs = torch.sin(2 * np.pi * self.foot_indices)
        self.doubletime_clock_inputs = torch.sin(4 * np.pi * self.foot_indices)
        self.halftime_clock_inputs = torch.sin(np.pi * self.foot_indices)

        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf
        for i in range(num_feet):
            idxs = self.foot_indices[:, i]
            smoothing_multiplier = (smoothing_cdf_start(torch.remainder(idxs, 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(idxs, 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(idxs, 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(idxs, 1.0) - 0.5 - 1)))
            self.desired_contact_states[:, i] = smoothing_multiplier

    def _update_feet_pos_local(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        self.desired_feet_pos_local[:, :, 2] = self.gait_feet_height * phases + 0.02

        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequency = self.gait_frequency
        x_vel_des = self.commands[:, 0:1]
        y_vel_des = self.commands[:, 1:2]
        yaw_vel_des = self.commands[:, 2:3]
        desired_xs_norm = self.gait_feet_stationary_pos[:, :, 0]
        desired_ys_norm = self.gait_feet_stationary_pos[:, :, 1]

        desired_xs_offset = phases * x_vel_des * (0.5 / frequency)
        desired_ys_offset = phases * y_vel_des * (0.5 / frequency)
        yaw_to_y_vel_des = yaw_vel_des * desired_xs_norm
        desired_yaw_to_ys_offset = phases * yaw_to_y_vel_des * (0.5 / frequency)
        yaw_to_x_vel_des = -yaw_vel_des * desired_ys_norm
        desired_yaw_to_xs_offset = phases * yaw_to_x_vel_des * (0.5 / frequency)

        self.desired_feet_pos_local[:, :, 0] = desired_xs_norm + (desired_xs_offset + desired_yaw_to_xs_offset)
        self.desired_feet_pos_local[:, :, 1] = desired_ys_norm + (desired_ys_offset + desired_yaw_to_ys_offset)

        center = self.body_pos.clone().unsqueeze(1)
        center[:, :, 2] = 0.0
        feet_pos_translated = self.foot_positions - center
        body_quat_rel = gs_quat_mul(self.body_quat, gs_inv_quat(self.body_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        for i in range(len(self.feet_link_indices)):
            self.feet_pos_local[:, i, :] = gs_quat_apply_yaw(gs_quat_conjugate(body_quat_rel), feet_pos_translated[:, i, :])

    def _update_buffers(self):
        super()._update_buffers()
        self.foot_positions[:] = self.robot.get_links_pos()[:, self.feet_link_indices]
        self.foot_quaternions[:] = self.robot.get_links_quat()[:, self.feet_link_indices]
        self.foot_velocities[:] = self.robot.get_links_vel()[:, self.feet_link_indices]

        self._update_desired_contact_states()
        self._update_feet_pos_local()

    def post_physics_step(self):
        super().post_physics_step()
        self._update_desired_contact_states()
        self._update_feet_pos_local()