
import numpy as np
import torch

import genesis as gs
from envs.vec_env import VecEnv
from utils import *

class StateEnv(VecEnv):
    def __init__(self, num_envs, env_cfg, show_viewer, eval, debug, n_rendered_envs=1, device='cuda'):
        super().__init__(num_envs, env_cfg, show_viewer, eval, debug, n_rendered_envs, device)

    def _init_buffers(self):
        super()._init_buffers()

        # Safely initialize feet_link_indices
        self.feet_link_indices = self._find_link_indices(
            self.env_cfg.get('feet_link_names', [])
        )

        # Initialize additional buffers for commands
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

        # Initialize additional buffers for foot-related data only if feet_link_indices is valid
        if self.feet_link_indices:
            self.foot_positions = torch.zeros(
                self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float
            )
            self.foot_quaternions = torch.zeros(
                self.num_envs, len(self.feet_link_indices), 4, device=self.device, dtype=gs.tc_float
            )
            self.foot_velocities = torch.zeros(
                self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float
            )

            # Current phase and desired gait
            self.gait_indices = torch.zeros(
                (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float
            )
            self.foot_indices = torch.zeros(
                (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float
            )
            self.gait_frequency = torch.ones(
                (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float
            )
            self.gait_duration = torch.zeros(
                (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float
            )
            self.gait_offset = torch.zeros(
                (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float
            )
            self.gait_feet_height = torch.zeros(
                (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float
            )
            self.gait_feet_stationary_pos = torch.zeros(
                (self.num_envs, len(self.feet_link_indices), 2), device=self.device, dtype=gs.tc_float
            )
            self.gait_body_height = torch.zeros(
                self.num_envs, device=self.device, dtype=gs.tc_float
            )

            # Time embed
            self.clock_inputs = torch.zeros(
                (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float
            )
            self.doubletime_clock_inputs = torch.zeros(
                (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float
            )
            self.halftime_clock_inputs = torch.zeros(
                (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float
            )

            # Reference buffer
            self.desired_contact_states = torch.zeros(
                (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float
            )
            self.desired_feet_pos_local = torch.zeros(
                (self.num_envs, len(self.feet_link_indices), 3), device=self.device, dtype=gs.tc_float
            )
            self.feet_pos_local = torch.zeros(
                (self.num_envs, len(self.feet_link_indices), 3), device=self.device, dtype=gs.tc_float
            )

            if 'gait' in self.env_cfg:
                self._load_gait(self.env_cfg['gait'])

    def _update_buffers(self):
        super()._update_buffers()

        if self.feet_link_indices:
            self.foot_positions[:] = self.robot.get_links_pos()[:, self.feet_link_indices]
            self.foot_quaternions[:] = self.robot.get_links_quat()[:, self.feet_link_indices]
            self.foot_velocities[:] = self.robot.get_links_vel()[:, self.feet_link_indices]
            self._update_desired_contact_states()
            self._update_feet_pos_local()

    def post_physics_step(self):
        super().post_physics_step()

    def compute_observation(self):
        obs_buf = torch.cat(
            [
                self.body_ang_vel_local * self.obs_scales['ang_vel'],               # 3
                self.projected_gravity,                                             # 3
                self.commands * self.commands_scale,                                # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
                self.dof_vel * self.obs_scales['dof_vel'],                         
                self.actions,
                self.clock_inputs if self.feet_link_indices else torch.zeros((self.num_envs, 0), device=self.device),
            ],
            axis=-1,
        )

        if not self.eval:
            obs_buf += gs_rand_float(-1.0, 1.0, (self.num_obs,), self.device) * self.obs_noise

        clip_obs = 100.0
        self.obs_buf = torch.clip(obs_buf, -clip_obs, clip_obs)

    def compute_critic_observation(self):
        privileged_obs_buf = torch.cat(
            [
                self.body_lin_vel * self.obs_scales['lin_vel'],                     # 3
                self.body_ang_vel * self.obs_scales['ang_vel'],                     # 3
                self.projected_gravity,                                             # 3
                self.commands * self.commands_scale,                                # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
                self.dof_vel * self.obs_scales['dof_vel'],
                self.actions,
                self.last_actions,
                self.clock_inputs if self.feet_link_indices else torch.zeros((self.num_envs, 0), device=self.device),
            ],
            axis=-1,
        )

        clip_obs = 100.0
        self.privileged_obs_buf = torch.clip(privileged_obs_buf, -clip_obs, clip_obs)

    def compute_state(self):
        self.state_buf = torch.cat(
            [
                self.base_pos[:, 2:],
                self.projected_gravity,
                self.base_lin_vel,
                self.base_ang_vel,
                self.dof_pos,
                self.dof_vel,
            ],
            axis=-1,
        )

    def _load_gait(self, gait_cfg):
        self.gait_cfg = gait_cfg
        self.gait_body_height[:] = gait_cfg.get('base_height_target', 0.0)
        for i in range(len(self.feet_link_indices)):
            self.gait_frequency[:, i] = gait_cfg.get('frequency', [0.0] * len(self.feet_link_indices))[i]
            self.gait_duration[:, i] = gait_cfg.get('duration', [0.0] * len(self.feet_link_indices))[i]
            self.gait_offset[:, i] = gait_cfg.get('offset', [0.0] * len(self.feet_link_indices))[i]
            self.gait_feet_height[:, i] = gait_cfg.get('feet_height_target', [0.0] * len(self.feet_link_indices))[i]
            self.gait_feet_stationary_pos[:, i, :] = torch.tensor(
                gait_cfg.get('stationary_position', [[0.0, 0.0]] * len(self.feet_link_indices))[i], 
                device=self.device
            )

    def _update_desired_contact_states(self):
        if not self.feet_link_indices:
            return

        num_feet = len(self.feet_link_indices)
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * self.gait_frequency, 1.0)        
        self.foot_indices = torch.remainder(self.gait_indices + self.gait_offset, 1.0)

        for i in range(num_feet):
            idxs = self.foot_indices[:, i]
            duration = self.gait_duration[:, i]
            stance_idxs = torch.remainder(idxs, 1) < duration
            swing_idxs = torch.remainder(idxs, 1) > duration

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / duration[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - duration[swing_idxs]) * (0.5 / (1 - duration[swing_idxs]))

        self.clock_inputs = torch.sin(2 * np.pi * self.foot_indices)
        self.doubletime_clock_inputs = torch.sin(4 * np.pi * self.foot_indices)
        self.halftime_clock_inputs = torch.sin(np.pi * self.foot_indices)

        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf  
        for i in range(num_feet):
            idxs = self.foot_indices[:, i]
            smoothing_multiplier = (smoothing_cdf_start(torch.remainder(idxs, 1.0)) * 
                                    (1 - smoothing_cdf_start(torch.remainder(idxs, 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(idxs, 1.0) - 1) * 
                                    (1 - smoothing_cdf_start(torch.remainder(idxs, 1.0) - 0.5 - 1)))
            self.desired_contact_states[:, i] = smoothing_multiplier
    
    def _update_feet_pos_local(self):
        if not self.feet_link_indices:
            return

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
        yaw_to_x_vel_des = - yaw_vel_des * desired_ys_norm
        desired_yaw_to_xs_offset = phases * yaw_to_x_vel_des * (0.5 / frequency)

        self.desired_feet_pos_local[:, :, 0] = desired_xs_norm + (desired_xs_offset + desired_yaw_to_xs_offset)
        self.desired_feet_pos_local[:, :, 1] = desired_ys_norm + (desired_ys_offset + desired_yaw_to_ys_offset)

        center = self.body_pos.clone().unsqueeze(1)
        center[:, :, 2] = 0.0
        feet_pos_translated = self.foot_positions - center
        body_quat_rel = gs_quat_mul(self.body_quat, gs_inv_quat(self.body_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        for i in range(len(self.feet_link_indices)):
            self.feet_pos_local[:, i, :] = gs_quat_apply_yaw(gs_quat_conjugate(body_quat_rel), feet_pos_translated[:, i, :])

    def _find_link_indices(self, names, accurate=True):
        link_indices = []
        available = [True for _ in range(len(self.robot.links))]
        for name in names:
            for i, link in enumerate(self.robot.links):
                if available[i] and (not accurate and name in link.name or name == link.name):
                    available[i] = False
                    link_indices.append(link.idx - self.robot.link_start)

        return link_indices