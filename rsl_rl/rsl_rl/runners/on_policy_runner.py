#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import wandb
import numpy as np
import math

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.utils import store_code_state


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.print_infos = self.cfg.get("print_infos", True)
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        self.env.reset()
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        iter_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # move to the right device
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs
                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        iter_infos.append(infos.copy())
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if self.cfg["record_interval"] > 0 and self.logger_type == "wandb":
                self.log_video()
                if self.cfg["record_interval"] > 0 and it % self.cfg["record_interval"] == 0:
                    self.start_recording()
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            iter_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        step = locs["it"]
        iter_infos = locs["iter_infos"]
        ep_string = ""
        rew_string = ""
        if len(iter_infos) > 0:
            if "episode" in iter_infos[0].keys():
                for key in iter_infos[0]["episode"].keys():
                    value_list = []
                    for iter_info in iter_infos:
                        if iter_info["episode"][key] is not None:
                            value_list.append(iter_info["episode"][key])
                    value = torch.mean(torch.tensor(value_list))
                    value = 0.0 if torch.isnan(value) else value
                    self.writer.add_scalar(f"Episode/{key}", value, step)
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
            if "rewards" in iter_infos[0].keys():
                for key in iter_infos[0]["rewards"].keys():
                    value_list = []
                    for iter_info in iter_infos:
                        value_list.append(iter_info["rewards"][key])
                    value = torch.mean(torch.tensor(value_list))
                    self.writer.add_scalar(f"StepRew/{key}", value, step)
                    rew_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
            if "log" in iter_infos[0].keys():
                for key in iter_infos[0]["log"].keys():
                    value_list = []
                    for iter_info in iter_infos:
                        value_list.append(iter_info["log"][key])
                    value = torch.mean(torch.tensor(value_list))
                    self.writer.add_scalar(f"Log/{key}", value, step)

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], step)
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], step)
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, step)
        self.writer.add_scalar("Loss/log_learning_rate", math.log(self.alg.learning_rate), step)
        self.writer.add_scalar("Loss/kl", self.alg.kl_mean, step)
        self.writer.add_scalar("Loss/mean_noise_std", mean_std.item(), step)
        self.writer.add_scalar("Perf/total_fps", fps, step)
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], step)
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], step)
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_episode_reward", statistics.mean(locs["rewbuffer"]), step)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), step)
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_episode_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        if self.print_infos:
            log_string += f"""{'-' * width}\n"""
            log_string += "Per Episode".center(width, " ") + "\n"
            log_string += ep_string
            log_string += f"""{'-' * width}\n"""
            log_string += "Per Step".center(width, " ") + "\n"
            log_string += rew_string

        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    def start_recording(self):
        self.env.start_recording()

    def log_video(self):
        frames = self.env.get_recorded_frames()
        if frames is None:
            return
        else:
            video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
            wandb.log({"video": wandb.Video(video_array, fps=int(1/self.env.dt))})