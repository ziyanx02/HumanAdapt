#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

import time

from rsl_rl.modules import ActorCriticTDO, TemporalDistribution
from rsl_rl.storage import TDORolloutStorage


class TDO:
    actor_critic: ActorCriticTDO
    temporal_distribution: TemporalDistribution

    def __init__(
        self,
        actor_critic,
        temporal_distribution,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        td_entropy_coef=0.0,
        return_boosting_coef=0.0,
        action_noise_threshold=1.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        is_PPO=False,
        device="cpu",
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.is_PPO = is_PPO

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = TDORolloutStorage.Transition()

        # TDO components
        self.temporal_distribution = temporal_distribution
        self.temporal_distribution.to(self.device)
        self.td_optimizer = optim.Adam(self.temporal_distribution.parameters(), lr=self.temporal_distribution.learning_rate)

        # TDO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.td_entropy_coef = td_entropy_coef
        self.return_boosting_coef = return_boosting_coef
        self.action_noise_threshold = action_noise_threshold

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, state_shape, period_length):
        self.storage = TDORolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, state_shape, period_length, self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def sample(self, time):
        return self.temporal_distribution.sample(time)

    def act(self, obs, critic_obs, times):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, times).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, state, phase):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.state = state
        self.transition.phase = phase
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        values_by_time = self.storage.values_by_time
        value_mean_by_time = values_by_time.mean(dim=(0, 2))
        value_std_by_time = values_by_time.std(dim=(0, 2)) + 1e-7
        state_mean_before_update = self.temporal_distribution.mean_params.data.clone().detach()
        mean_value_loss = 0
        mean_surrogate_loss = 0
        generator = self.storage.tdo_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            states_batch,
            phases_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, phases_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)
                self.kl_mean = kl_mean

                if self.desired_kl is not None and self.schedule == "adaptive":
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.ac_optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.ac_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.ac_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            if self.is_PPO or sigma_batch.mean().item() > self.action_noise_threshold:
                continue

            states_log_prob_batch = self.temporal_distribution.get_states_log_prob(states_batch, phases_batch)
            states_entropy_batch = self.temporal_distribution.entropy

            # Transition loss
            transition_loss = -states_log_prob_batch.mean()

            # Return boosting loss
            ratio = (returns_batch - value_mean_by_time[phases_batch]) / value_std_by_time[phases_batch]
            return_boosting_loss = (ratio * states_log_prob_batch).mean()

            loss = transition_loss + self.return_boosting_coef * return_boosting_loss + self.td_entropy_coef * states_entropy_batch.mean()

            self.td_optimizer.zero_grad()
            transition_loss.backward()
            nn.utils.clip_grad_norm_(self.temporal_distribution.parameters(), self.max_grad_norm)
            self.td_optimizer.step()

        state_mean_after_update = self.temporal_distribution.mean_params.data.clone().detach()
        state_mean_difference = torch.abs(state_mean_after_update - state_mean_before_update)
        mean_state_update = state_mean_difference.mean().item()
        max_state_update = state_mean_difference.max().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_state_update, max_state_update
