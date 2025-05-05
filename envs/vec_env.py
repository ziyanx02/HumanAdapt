import numpy as np
import torch
import math

import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from utils import *

def quaternion_from_projected_gravity(gravity):
    """
    Compute the minimum quaternion to rotate v1 to v2 for batched tensors.
    v1: Target batched vectors, shape (N, 3)
    v2: Fixed vector (0, 0, -1), shape (3,)
    Returns: Quaternions of shape (N, 4) (w, x, y, z)
    """
    # Normalize gravity vectors
    v1 = torch.nn.functional.normalize(gravity, dim=-1)

    # Define fixed vector
    v2 = torch.zeros_like(gravity, dtype=gravity.dtype, device=gravity.device)
    v2[..., 2] = -1.0

    # Compute dot product and angle
    dot = torch.sum(v1 * v2, dim=-1, keepdim=True)  # Shape (N, 1)
    angle = torch.acos(torch.clamp(dot, -1.0, 1.0))  # Clamp to avoid NaN due to precision errors
    
    # Compute rotation axis
    axis = torch.cross(v1.expand_as(v2), v2, dim=-1)  # Shape (N, 3)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    
    # Handle parallel vectors (zero cross product)
    axis = torch.where(axis_norm > 1e-6, axis / axis_norm, torch.tensor([1.0, 0.0, 0.0], device=v2.device))
    
    # Compute quaternion components
    w = torch.cos(angle / 2)
    xyz = axis * torch.sin(angle / 2)
    
    return torch.cat([w, xyz], dim=-1)  # Shape (N, 4)

class VecEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        show_viewer,
        eval,
        debug,
        n_rendered_envs=1,
        device='cuda',
    ) -> None:
        self.num_envs = num_envs
        self.cfg = env_cfg
        self.env_cfg = env_cfg
        self.num_actions = env_cfg['num_actions']
        self.num_states = env_cfg['num_states']
        self.num_dof = env_cfg['num_dofs']
        self.obs_cfg = env_cfg['observation']
        self.delay_action = env_cfg['delay_action']
        self.is_PPO = env_cfg['PPO']
        self.dt = 1 / env_cfg['control_freq']
        self.max_episode_length_s = env_cfg['episode_length_s']
        self.period_length_s = env_cfg['period_length_s']
        self.obs_scales = self.obs_cfg['obs_scales']
        self.num_obs = self.obs_cfg['num_obs']
        self.num_privileged_obs = self.obs_cfg['num_priv_obs']
        self.reward_cfg = env_cfg['reward']
        self.reward_scales = self.reward_cfg['reward_scales']
        self.max_episode_length = int(np.ceil(self.max_episode_length_s / self.dt))
        self.period_length = int(np.ceil(self.period_length_s / self.dt))
        self.record_length = int(np.ceil(env_cfg.get('record_length', 3) / self.dt))

        self.headless = not show_viewer
        self.eval = eval
        self.debug = debug
        self.n_rendered_envs = n_rendered_envs

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            assert device == 'cpu' or device.startswith('cuda')
            self.device = torch.device(device)

        self.scale = env_cfg.get('robot_scale', 1.0)
        if self.scale != 1.0:
            self.env_cfg['command']['lin_vel_x_range'] = [x * self.scale for x in self.env_cfg['command']['lin_vel_x_range']]
            self.env_cfg['command']['lin_vel_y_range'] = [x * self.scale for x in self.env_cfg['command']['lin_vel_y_range']]
            self.env_cfg['base_init_pos'] = [x * self.scale for x in self.env_cfg['base_init_pos']]
            if 'base_reset_pos' in self.env_cfg.keys():
                self.env_cfg['base_reset_pos'] = [x * self.scale for x in self.env_cfg['base_reset_pos']]
            for key in self.env_cfg['PD_stiffness'].keys():
                self.env_cfg['PD_stiffness'][key] *= self.scale
            for key in self.env_cfg['PD_damping'].keys():
                self.env_cfg['PD_damping'][key] *= self.scale
            self.env_cfg['termination_if_height_lower_than'] *= self.scale
            self.env_cfg['gait']['base_height_target'] *= self.scale
            self.env_cfg['gait']['feet_height_target'] = [x * self.scale for x in self.env_cfg['gait']['feet_height_target']]
            self.env_cfg['gait']['stationary_position'] = [[x[0] * self.scale, x[1] * self.scale] for x in self.env_cfg['gait']['stationary_position']]

        self._create_scene()
        if gs.platform != 'macOS':
            self._set_camera()
        self.scene.build(n_envs=num_envs)
        self._init_buffers()
        self._prepare_reward_function()
        self._domain_randomization()

    def _create_scene(self):
        sim_dt = self.dt / self.env_cfg['decimation']
        sim_substeps = 1

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=sim_dt,
                substeps=sim_substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.env_cfg['decimation']),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                n_rendered_envs=self.n_rendered_envs,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=not self.headless,
            show_FPS=False,
        )

        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

        self.scene.add_entity(
            gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True),
        )

        self.base_init_pos = torch.tensor(
            self.env_cfg['base_init_pos'], device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg['base_init_quat'], device=self.device
        )

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.env_cfg['urdf_path'],
                merge_fixed_links=True,
                links_to_keep=self.env_cfg['links_to_keep'],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                scale=self.scale,
            ),
            visualize_contact=self.debug,
        )

    def _prepare_reward_function(self):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            for name in self.reward_scales.keys()
        }

    def _init_buffers(self):
        self.base_euler = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.global_gravity = torch.tensor(
            np.array([0.0, 0.0, -1.0]), device=self.device, dtype=gs.tc_float
        )
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.forward_vec[:, 0] = 1.0

        self.state_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=gs.tc_float
        )
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self.obs_noise = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self._prepare_obs_noise()

        self.privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros(
                (self.num_envs, self.num_privileged_obs),
                device=self.device,
                dtype=gs.tc_float,
            )
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf_pos = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf_neg = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.terminate_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.time_out_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )

        # names to indices
        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local
            for name in self.env_cfg['dof_names']
        ]

        def find_link_indices(names, accurate=False):
            link_indices = list()
            availible = [True for i in range(len(self.robot.links))]
            for name in names:
                for i, link in enumerate(self.robot.links):
                    if availible[i] and (accurate==False and name in link.name or name == link.name):
                        availible[i] = False
                        link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.termination_contact_link_indices = find_link_indices(
            self.env_cfg['termination_contact_link_names']
        )
        self.penalized_contact_link_indices = find_link_indices(
            self.env_cfg['penalized_contact_link_names']
        )
        assert len(self.termination_contact_link_indices) > 0
        assert len(self.penalized_contact_link_indices) > 0

        # actions
        self.actions = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.last_last_actions = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.dof_pos = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.dof_vel = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.last_dof_vel = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.root_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.last_root_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )

        self.base_reset_pos = torch.tensor(
            self.env_cfg.get('base_reset_pos', self.env_cfg['base_init_pos']), device=self.device
        )
        self.base_reset_quat = torch.tensor(
            self.env_cfg.get('base_reset_quat', self.env_cfg['base_init_quat']), device=self.device
        )
        self.projected_gravity_reset = gs_transform_by_quat(
            self.global_gravity, gs_inv_quat(gs_quat_mul(self.base_reset_quat, gs_inv_quat(self.base_init_quat)))
        )

        # body
        self.base_link_index = 0
        self.body_link_index = find_link_indices(self.env_cfg['body_link_name'], accurate=True)
        if self.body_link_index[0] != 0:
            self.body_init_pos = torch.tensor(
                self.env_cfg['body_init_pos'], device=self.device
            )
            self.body_init_quat = torch.tensor(
                self.env_cfg['body_init_quat'], device=self.device
            )   
        else:
            self.body_init_pos = self.base_init_pos
            self.body_init_quat = self.base_init_quat

        self.body_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.body_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.body_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.body_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.body_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.body_ang_vel_local = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        ) # For real deploy, apply full rotation on ang_vel instead of only yaw rotation as `self.body_ang_vel` has
        self.body_projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.com = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=gs.tc_float,
        )

        # extras
        self.continuous_push = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int, 
        )
        self.common_step_counter = 0
        self.extras = {"log": {}}
        self.extras["observations"] = {"critic": self.privileged_obs_buf}

        # PD control
        stiffness = self.env_cfg['PD_stiffness']
        damping = self.env_cfg['PD_damping']

        self.p_gains, self.d_gains = [], []
        for dof_name in self.env_cfg['dof_names']:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device, dtype=gs.tc_float)
        self.d_gains = torch.tensor(self.d_gains, device=self.device, dtype=gs.tc_float)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)

        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)

        default_joint_angles = self.env_cfg['default_joint_angles']
        self.default_dof_pos = torch.tensor(
            [default_joint_angles[name] for name in self.env_cfg['dof_names']],
            device=self.device,
        )
        reset_joint_angles = self.env_cfg.get('reset_joint_angles', self.env_cfg['default_joint_angles'])
        self.reset_dof_pos = torch.tensor(
            [reset_joint_angles[name] for name in self.env_cfg['dof_names']],
            device=self.device,
        )

        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.dof_pos_limits = torch.clip(self.dof_pos_limits, -torch.pi * 2, torch.pi * 2)
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1] * pow(self.scale, 5)

        self._prepare_temporal_distribution()

        self.motor_strengths = torch.ones((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.motor_offsets = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)

    def _prepare_temporal_distribution(self):
        init_state_mean = torch.cat(
            [
                self.base_reset_pos[2:],
                self.projected_gravity_reset,
                torch.zeros((3,), device=self.device, dtype=gs.tc_float),
                torch.zeros((3,), device=self.device, dtype=gs.tc_float),
                self.reset_dof_pos,
                torch.zeros((self.num_dof,), device=self.device, dtype=gs.tc_float),
            ],
            axis=-1,
        )
        init_state_std = torch.cat(
            [
                0.0 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((3,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((3,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((3,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((self.num_dof,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((self.num_dof,), device=self.device, dtype=gs.tc_float),
            ],
            axis=-1,
        )
        if self.eval:
            init_state_std *= 0.0
        self.state_mean = init_state_mean.unsqueeze(0).repeat(self.period_length, 1)
        self.state_std = init_state_std.unsqueeze(0).repeat(self.period_length, 1)
        self.init_state_min = torch.cat(
            [
                0.2 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                -torch.ones((3,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((3,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((3,), device=self.device, dtype=gs.tc_float),
                self.dof_pos_limits[:, 0],
                0.0 * torch.ones((self.num_dof,), device=self.device, dtype=gs.tc_float),
            ],
            axis=-1,
        )
        self.init_state_max = torch.cat(
            [
                0.5 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                torch.ones((3,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((3,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((3,), device=self.device, dtype=gs.tc_float),
                self.dof_pos_limits[:, 1],
                0.0 * torch.ones((self.num_dof,), device=self.device, dtype=gs.tc_float),
            ],
            axis=-1,
        )
        self.link_contact_forces_limit = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=self.device
        )

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return None, None

    def step(self, actions):
        clip_actions = self.env_cfg['clip_actions']
        self.actions = torch.clip(actions, -clip_actions, clip_actions) * 0
        exec_actions = self.last_actions if self.delay_action else self.actions

        for _ in range(self.env_cfg['decimation']):
            self.torques = self._compute_torques(exec_actions)
            self.robot.control_dofs_force(self.torques, self.motor_dofs)
            self.scene.step()
            self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
            self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        self.post_physics_step()

        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        init_states = self.state_mean[0] + gs_rand_float(
            -1.0, 1.0, (len(envs_idx), self.num_states), self.device
        ) * self.state_std[0]
        self.set_state(init_states, 0, envs_idx)

    def _compute_torques(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.env_cfg['action_scale']
        torques = (
            self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets)
            - self.batched_d_gains * self.dof_vel
        )
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        # self.extras["log"]["mean_torque"] = torques.abs().mean().item()
        # self.extras["log"]["max_torque"] = torques.abs().max().item()
        return torques * self.motor_strengths

    def _compute_target_dof_pos(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.env_cfg['action_scale']
        target_dof_pos = actions_scaled + self.default_dof_pos
        return target_dof_pos

    def post_physics_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._update_buffers()
        self.check_termination()
        self.compute_reward()
        self.compute_critic_observation()

        resampling_time_s = self.env_cfg['resampling_time_s']
        envs_idx = (
            (self.episode_length_buf % int(resampling_time_s / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._randomize_rigids(envs_idx)
        self._randomize_controls(envs_idx)

        # random push
        push_interval_s = self.env_cfg['push_interval_s']
        if push_interval_s > 0 and not (self.debug or self.eval):
            max_push_vel_xy = self.env_cfg['max_push_vel_xy']
            dofs_vel = self.robot.get_dofs_velocity() # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            push_vel[((self.common_step_counter + self.env_identities) % int(push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += push_vel
            self.robot.set_dofs_velocity(dofs_vel)

        envs_idx = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.update_extras(envs_idx)
        if self.is_PPO:
            self.reset_idx(envs_idx)
            self.compute_observation()
        self.compute_state()

        if gs.platform != 'macOS':
            self._render_headless()
        if (self._recording or not self.headless) and self.debug:
            self._draw_debug_vis()

        self.last_actions[:] = self.actions[:]
        self.last_last_actions[:] = self.last_actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.robot.get_vel()

    # ------------ update buffers ----------------

    def _update_buffers(self):

        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        base_quat_rel = gs_quat_mul(self.base_quat, gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.base_euler = gs_quat2euler(base_quat_rel)

        inv_quat_yaw = gs_quat_from_angle_axis(-self.base_euler[:, 2],
                                               torch.tensor([0, 0, 1], device=self.device, dtype=torch.float))

        self.base_lin_vel[:] = gs_transform_by_quat(self.robot.get_vel(), inv_quat_yaw)
        self.base_ang_vel[:] = gs_transform_by_quat(self.robot.get_ang(), inv_quat_yaw)
        self.projected_gravity = gs_transform_by_quat(
            self.global_gravity, gs_inv_quat(base_quat_rel)
        )

        # Same as base
        self.body_pos[:] = self.robot.get_links_pos()[:, self.body_link_index].squeeze(1)
        self.body_quat[:] = self.robot.get_links_quat()[:, self.body_link_index].squeeze(1)
        body_quat_rel = gs_quat_mul(self.body_quat, gs_inv_quat(self.body_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.body_euler = gs_quat2euler(body_quat_rel)

        inv_quat_yaw = gs_quat_from_angle_axis(-self.body_euler[:, 2],
                                               torch.tensor([0, 0, 1], device=self.device, dtype=torch.float))

        self.body_lin_vel[:] = gs_transform_by_quat(self.robot.get_links_vel()[:, self.body_link_index].squeeze(1), inv_quat_yaw)
        self.body_ang_vel[:] = gs_transform_by_quat(self.robot.get_links_ang()[:, self.body_link_index].squeeze(1), inv_quat_yaw)
        self.body_ang_vel_local[:] = gs_transform_by_quat(self.robot.get_links_ang()[:, self.body_link_index].squeeze(1), gs_inv_quat(body_quat_rel))
        self.body_projected_gravity = gs_transform_by_quat(
            self.global_gravity, gs_inv_quat(body_quat_rel)
        )
        # print(self.body_projected_gravity)
        # print(gs_transform_by_quat(gs_transform_by_quat(self.global_gravity, gs_inv_quat(self.base_quat)), self.base_init_quat))

        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.link_contact_forces[:] = self.robot.get_links_net_contact_force()
        self.com[:] = self.rigid_solver.get_links_COM([self.base_link_index,]).squeeze(dim=1)

    def check_termination(self):

        self.terminate_buf = torch.any(
            torch.norm(
                self.link_contact_forces[:, self.termination_contact_link_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )
        self.terminate_buf |= torch.any(
            torch.norm(
                self.link_contact_forces[:, self.penalized_contact_link_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        ) & (self.episode_length_buf > 2.0 / self.dt)

        if self.debug:
            link_contact = torch.norm(self.link_contact_forces[:, :, :], dim=-1,) > 1.0
            link_contact_name = []
            for i in range(link_contact.shape[1]):
                if link_contact[0][i] == True:
                    link_contact_name.append(self.robot.links[i].name)
            print(f'Contact link names: {link_contact_name}')

        self.terminate_buf |= torch.logical_or(
            torch.abs(self.base_euler[:, 1])
            > self.env_cfg['termination_if_pitch_greater_than'],
            torch.abs(self.base_euler[:, 0])
            > self.env_cfg['termination_if_roll_greater_than'],
        ) & (self.episode_length_buf > 1.0 / self.dt)
        
        if self.debug and self.terminate_buf.any():
            contact = torch.any(torch.norm(self.link_contact_forces[:, self.termination_contact_link_indices, :], dim=-1,)> 1.0,dim=1,)
            pitch = torch.abs(self.base_euler[:, 1]) > self.env_cfg['termination_if_pitch_greater_than']
            roll = torch.abs(self.base_euler[:, 0]) > self.env_cfg['termination_if_roll_greater_than']
            print(f'Contact {contact}\nPitch {pitch}\nRoll {roll}')
            if contact.any():
                contact_idx = torch.norm(self.link_contact_forces[:, self.termination_contact_link_indices, :], dim=-1,) > 1.0
                print(f'Detail contact: {contact_idx}')
            if pitch.any():
                print(f'Pitch angle: {self.base_euler[:, 1]}')
            if roll.any():
                print(f'Roll angle: {self.base_euler[:, 0]}')

        self.terminate_buf |= self.base_pos[:, 2] < self.env_cfg['termination_if_height_lower_than']
        if self.env_cfg['use_timeout']:
            self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = torch.logical_or(self.terminate_buf, self.time_out_buf)

    def compute_state(self):
        '''
        State for restoring and observations: 
        - height # 1
        - projected_gravity # 3
            apply the reverse rotation to vector of global gravity to obtain projected gravity (i.e. gravity vector in robot local frame)
        - base_lin_vel # 3 
            apply yaw rotation to the global lin_vel of base link, i.e. only transform at heading direction
        - base_ang_vel # 3 
            apply yaw rotation to the global ang_vel of base link, i.e. only transform at heading direction
        - dof_pos # num_dof
        - dof_vel # num_dof
        '''
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

    def _prepare_obs_noise(self):
        self.obs_noise[:3] = self.obs_cfg['obs_noise']['ang_vel']
        self.obs_noise[3:6] = self.obs_cfg['obs_noise']['gravity']
        self.obs_noise[9:21] = self.obs_cfg['obs_noise']['dof_pos']
        self.obs_noise[21:33] = self.obs_cfg['obs_noise']['dof_vel']

    def compute_observation(self):
        obs_buf = torch.cat(
            [
                self.body_ang_vel_local * self.obs_scales['ang_vel'],               # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
                self.dof_vel * self.obs_scales['dof_vel'],                         
                self.actions,
            ],
            axis=-1,
        )
        # add noise
        if not self.eval:
            obs_buf += gs_rand_float(
                -1.0, 1.0, (self.num_obs,), self.device
            )  * self.obs_noise

        clip_obs = 100.0
        self.obs_buf = torch.clip(obs_buf, -clip_obs, clip_obs)

    def compute_critic_observation(self):
        privileged_obs_buf = torch.cat(
            [
                self.body_lin_vel * self.obs_scales['lin_vel'],                     # 3
                self.body_ang_vel * self.obs_scales['ang_vel'],                     # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
                self.dof_vel * self.obs_scales['dof_vel'],
                self.actions,
                self.last_actions,
            ],
            axis=-1,
        )
        clip_obs = 100.0
        self.privileged_obs_buf = torch.clip(privileged_obs_buf, -clip_obs, clip_obs)

    def compute_reward(self):
        self.rew_buf[:] = 0.
        self.extras['rewards'] = {}
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.extras['rewards'][name] = torch.mean(rew).item()

    def update_extras(self, reset_envs_idx):

        # fill extras
        self.extras['episode'] = {}
        for key in self.episode_sums.keys():
            mean_episode_sum = torch.mean(self.episode_sums[key][reset_envs_idx]).item()
            self.extras['episode'][key] = None if math.isnan(mean_episode_sum) else mean_episode_sum
            self.episode_sums[key][reset_envs_idx] = 0.0

        if self.env_cfg['use_timeout']:
            self.extras['time_outs'] = self.time_out_buf

    def get_state(self):
        return self.state_buf, self.episode_length_buf

    def get_observations(self):
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def set_state(self, states, times, envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self.device)
        if len(envs_idx) == 0:
            return

        self.episode_length_buf[envs_idx] = times
        self._set_state(states, envs_idx)

    def _set_state(self, states, envs_idx):
        z = states[:, 0]
        projected_gravity = states[:, 1:4]
        lin_vel = states[:, 4:7]
        ang_vel = states[:, 7:10]
        dof_pos = states[:, 10:10+self.num_dof]
        dof_vel = states[:, 10+self.num_dof:10+2*self.num_dof]

        # reset dofs
        self.dof_pos[envs_idx] = dof_pos
        self.dof_vel[envs_idx] = dof_vel
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=False,
            envs_idx=envs_idx,
        )
        self.robot.set_dofs_velocity(
            velocity=self.dof_vel[envs_idx],
            dofs_idx_local=self.motor_dofs,
            envs_idx=envs_idx,
        )

        # update projected gravity
        base_quat_rel = gs_quat_mul(self.base_quat[envs_idx], gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(len(envs_idx), 1)))
        self.projected_gravity[envs_idx] = gs_transform_by_quat(
            self.global_gravity, gs_inv_quat(base_quat_rel)
        )

        # reset root states - position
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_pos[envs_idx, :2] += gs_rand_float(
            -1.0, 1.0, (len(envs_idx), 2), self.device
        ) * self.pos_init_randomization_scale
        self.base_pos[envs_idx, 2] = z
        self.robot.set_pos(
            self.base_pos[envs_idx],
            zero_velocity=False,
            envs_idx=envs_idx,
        )

        # reset root states - orientation
        rotation = quaternion_from_projected_gravity(projected_gravity)
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.base_quat[envs_idx] = gs_quat_mul(
            rotation,
            self.base_quat[envs_idx],
        )
        self.robot.set_quat(
            self.base_quat[envs_idx],
            zero_velocity=False,
            envs_idx=envs_idx
        )

        # reset root velocity
        self.base_lin_vel[envs_idx] = lin_vel
        self.base_ang_vel[envs_idx] = ang_vel
        base_vel = torch.concat(
            [gs_quat_apply_yaw(gs_quat_conjugate(rotation), self.base_lin_vel[envs_idx]), 
             gs_quat_apply_yaw(gs_quat_conjugate(rotation), self.base_ang_vel[envs_idx])], dim=1
        )
        self.robot.set_dofs_velocity(
            velocity=base_vel,
            dofs_idx_local=[0, 1, 2, 3, 4, 5],
            envs_idx=envs_idx,
        )

        # reset buffers
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.reset_buf[envs_idx] = 1

    # ------------ domain randomization ----------------

    def _domain_randomization(self):
        self._randomize_controls()
        self._randomize_rigids()
        self.pos_init_randomization_scale = self.env_cfg['pos_randomization']
        self.rot_init_randomization_scale = self.env_cfg['rot_randomization']
        self.dof_pos_init_randomization_scale = self.env_cfg['dof_pos_randomization']

    def _randomize_rigids(self, env_ids=None):

        if self.eval:
            return

        if env_ids == None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg['randomize_friction']:
            self._randomize_link_friction(env_ids)
        if self.env_cfg['randomize_base_mass']:
            self._randomize_base_mass(env_ids)
        if self.env_cfg['randomize_com_displacement']:
            self._randomize_com_displacement(env_ids)

        self.robot.set_dofs_damping([self.env_cfg['dof_damping'],] * self.num_dof, dofs_idx_local=self.motor_dofs)
        self.robot.set_dofs_armature([self.env_cfg['armature'],] * self.num_dof, dofs_idx_local=self.motor_dofs)

    def _randomize_controls(self, env_ids=None):

        if self.eval:
            return

        if env_ids == None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg['randomize_motor_strength']:
            self._randomize_motor_strength(env_ids)
        if self.env_cfg['randomize_motor_offset']:
            self._randomize_motor_offset(env_ids)
        if self.env_cfg['randomize_kp_scale']:
            self._randomize_kp(env_ids)
        if self.env_cfg['randomize_kd_scale']:
            self._randomize_kd(env_ids)

    def _randomize_link_friction(self, env_ids):

        min_friction, max_friction = self.env_cfg['friction_range']

        solver = self.rigid_solver

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) \
                 * (max_friction - min_friction) + min_friction
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)

    def _randomize_base_mass(self, env_ids):

        min_mass, max_mass = self.env_cfg['added_mass_range']
        base_link_id = 1

        added_mass = gs.rand((len(env_ids), 1), dtype=float) \
                        * (max_mass - min_mass) + min_mass

        self.rigid_solver.set_links_mass_shift(added_mass, [base_link_id,], env_ids)

    def _randomize_com_displacement(self, env_ids):

        min_displacement, max_displacement = self.env_cfg['com_displacement_range']
        base_link_id = 1

        com_displacement = torch.rand((len(env_ids), 1, 3), dtype=torch.float, device=self.device) \
                            * (max_displacement - min_displacement) + min_displacement
        # com_displacement[:, :, 0] -= 0.02

        self.rigid_solver.set_links_COM_shift(com_displacement, [base_link_id,], env_ids)

    def _randomize_motor_strength(self, env_ids):

        min_strength, max_strength = self.env_cfg['motor_strength_range']
        self.motor_strengths[env_ids, :] = torch.rand((len(env_ids), 1), dtype=torch.float, device=self.device) \
                                           * (max_strength - min_strength) + min_strength

    def _randomize_motor_offset(self, env_ids):

        min_offset, max_offset = self.env_cfg['motor_offset_range']
        self.motor_offsets[env_ids, :] = torch.rand((len(env_ids), self.num_dof), dtype=torch.float, device=self.device) \
                                         * (max_offset - min_offset) + min_offset

    def _randomize_kp(self, env_ids):

        min_scale, max_scale = self.env_cfg['kp_scale_range']
        kp_scales = torch.rand((len(env_ids), self.num_dof), dtype=torch.float, device=self.device) \
                    * (max_scale - min_scale) + min_scale
        self.batched_p_gains[env_ids, :] = kp_scales * self.p_gains[None, :]

    def _randomize_kd(self, env_ids):

        min_scale, max_scale = self.env_cfg['kd_scale_range']
        kd_scales = torch.rand((len(env_ids), self.num_dof), dtype=torch.float, device=self.device) \
                    * (max_scale - min_scale) + min_scale
        self.batched_d_gains[env_ids, :] = kd_scales * self.d_gains[None, :]

    # ------------ visualization ----------------

    def _draw_debug_vis(self):
        ''' Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        '''
        self.scene.clear_debug_objects()

        com = self.com[0]

        com[2] = 0.02
        self.scene.draw_debug_sphere(pos=com, radius=0.02, color=(0, 0, 1, 0.7))

    def _set_camera(self):
        ''' Set camera position and direction
        '''
        self._floating_camera = self.scene.add_camera(
            pos=np.array([0, -1.5, 1.5]),
            lookat=np.array([0, 0, 0]),
            # res=(720, 480),
            fov=50,
            GUI=False,
        )

        self._recording = False
        self._recorded_frames = []

        self.camera_x = self.scene.add_camera(
            pos=np.array([3, 0, 0.5]),
            lookat=np.array([0, 0, 0.5]),
            # res=(720, 480),
            fov=40,
            GUI=False,
        )
        self.camera_y = self.scene.add_camera(
            pos=np.array([0, 3, 0.5]),
            lookat=np.array([0, 0, 0.5]),
            # res=(720, 480),
            fov=40,
            GUI=False,
        )
        self.camera_z = self.scene.add_camera(
            pos=np.array([0, 0, 3]),
            lookat=np.array([0, 0, 0]),
            # res=(720, 480),
            fov=40,
            GUI=False,
        )

    def render_headless(self):
        return self.camera_x.render()[0], self.camera_y.render()[0], self.camera_z.render()[0]

    def _render_headless(self):
        if self._recording and len(self._recorded_frames) < self.record_length:
            robot_pos = np.array(self.base_pos[0].cpu())
            self._floating_camera.set_pose(pos=robot_pos + np.array([-1, -1, 0.5]), lookat=robot_pos + np.array([0, 0, -0.1]))
            # import time
            # start = time.time()
            frame = self._floating_camera.render()[0]
            # end = time.time()
            # print(end-start)
            self._recorded_frames.append(frame)
            # from PIL import Image
            # img = Image.fromarray(np.uint8(frame))
            # img.save('./test.png')
            # print('save')

    def get_recorded_frames(self):
        if len(self._recorded_frames) == self.record_length:
            frames = self._recorded_frames
            self._recorded_frames = []
            self._recording = False
            return frames
        else:
            return None

    def start_recording(self, record_internal=True):
        self._recorded_frames = []
        self._recording = True
        if record_internal:
            self._record_frames = True
        else:
            self._floating_camera.start_recording()

    def stop_recording(self, save_path=None):
        self._recorded_frames = []
        self._recording = False
        if save_path is not None:
            print("fps", int(1 / self.dt))
            self._floating_camera.stop_recording(save_path, fps = int(1 / self.dt))
