from typing import Dict, Sequence, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn as nn

from Humanoid_MARL.models.lstm import LSTM
from Humanoid_MARL.models.mlp import MLP


class AgentLSTM(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(
        self,
        policy_layers: Sequence[int],
        value_layers: Sequence[int],
        entropy_cost: float,
        discounting: float,
        reward_scaling: float,
        device: str,
        network_config: Dict = {},
    ):
        super(AgentLSTM, self).__init__()
        self.policy = LSTM(
            policy_layers,
            network_config["HIDDEN_SIZE"],
            num_layers=network_config["NUM_LAYERS"],
        ).to(device)
        self.value = MLP.construct(value_layers).to(device)

        self.num_steps = torch.zeros((), device=device)
        self.running_mean = torch.zeros(policy_layers[0], device=device)
        self.running_variance = torch.zeros(policy_layers[0], device=device)

        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = 0.95
        self.epsilon = 0.2
        self.device = device

    @torch.jit.export
    def dist_create(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normal followed by tanh.

        torch.distribution doesn't work with torch.jit, so we roll our own."""
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        scale = F.softplus(scale) + 0.001
        return loc, scale

    @torch.jit.export
    def dist_sample_no_postprocess(
        self, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return torch.normal(loc, scale)

    @classmethod
    def dist_postprocess(cls, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @torch.jit.export
    def dist_entropy(self, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        entropy = 0.5 + log_normalized
        entropy = entropy * torch.ones_like(loc)
        dist = torch.normal(loc, scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        entropy = entropy + log_det_jacobian
        return entropy.sum(dim=-1)

    @torch.jit.export
    def dist_log_prob(
        self, loc: torch.Tensor, scale: torch.Tensor, dist: torch.Tensor
    ) -> torch.Tensor:
        log_unnormalized = -0.5 * ((dist - loc) / scale).square()
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        log_prob = log_unnormalized - log_normalized - log_det_jacobian
        return log_prob.sum(dim=-1)

    @torch.jit.export
    def update_normalization(self, observation: torch.Tensor) -> None:
        self.num_steps += observation.shape[0] * observation.shape[1]
        input_to_old_mean = observation - self.running_mean
        mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
        self.running_mean = self.running_mean + mean_diff
        input_to_new_mean = observation - self.running_mean
        var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
        self.running_variance = self.running_variance + var_diff

    @torch.jit.export
    def normalize(self, observation: torch.Tensor) -> torch.Tensor:
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = torch.clip(variance, 1e-6, 1e6)
        return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

    @torch.jit.export
    def get_logits_action(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # observation = self.normalize(
        #     observation
        # )  # TODO: double check that normalization works with a LSTM
        logits = self.policy(observation)
        loc, scale = self.dist_create(logits)
        action = self.dist_sample_no_postprocess(loc, scale)
        return logits, action

    @torch.jit.export
    def compute_gae(
        self,
        truncation: torch.Tensor,
        termination: torch.Tensor,
        reward: torch.Tensor,
        values: torch.Tensor,
        bootstrap_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = (
            reward + self.discounting * (1 - termination) * values_t_plus_1 - values
        )
        deltas *= truncation_mask

        acc = torch.zeros_like(bootstrap_value)
        vs_minus_v_xs = torch.zeros_like(truncation_mask)

        for ti in range(truncation_mask.shape[0]):
            ti = truncation_mask.shape[0] - ti - 1
            acc = (
                deltas[ti]
                + self.discounting
                * (1 - termination[ti])
                * truncation_mask[ti]
                * self.lambda_
                * acc
            )
            vs_minus_v_xs[ti] = acc

        # Add V(x_s) to get v_s.
        vs = vs_minus_v_xs + values
        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)
        advantages = (
            reward + self.discounting * (1 - termination) * vs_t_plus_1 - values
        ) * truncation_mask
        return vs, advantages

    @torch.jit.export
    def loss(self, td: Dict[str, torch.Tensor], agent_idx: int) -> torch.Tensor:
        td_obs = td["observation"][:, :, :, agent_idx, :][
            :-1
        ]  # remove last observation, -> [unroll_length, batch_size, seq_len, obs_dim] : [1, 256, 4, 277]
        td_obs = td_obs.reshape(
            [-1] + list(td_obs.shape[-2:])
        )  # squeeze all dimensions except the last two -> [unroll_length * batch_size, seq_len, obs_dim] (to work with lstm)
        td_obs = td_obs.swapaxes(0, 1)  # [seq_len, batch_size, obs_dim] : [4, 256, 277]
        # ================= Getting Policy Logits =================
        policy_logits = self.policy(
            td_obs.swapaxes(0, 1)
        )  # [batch_dim, action_dim] : [256, 34]
        # ================ Getting Baseline =================
        baseline = self.value(td_obs)  # [seq_len, batch_dim, 1] : [4, 256, 1]
        baseline = torch.squeeze(baseline, dim=-1)  # [seq_len, batch_dim] : [4, 256]
        bootstrap_value = baseline[-1]  # [batch_dim] : [256]
        baseline = baseline[:-1]  # [seq_len - 1, batch_dim] : [3, 256]
        # ================= Getting Reward, Termination, Action =================
        reward = td["reward"][:, :, agent_idx] * self.reward_scaling  # [1, 256]
        termination = td["done"] * (1 - td["truncation"])  # [1, 256]
        action = td["action"].reshape(
            td["action"].shape[0],
            td["action"].shape[1],
            td["observation"].shape[-2],
            -1,
        )  # [1, 256, 2, 17]
        action_agent_idx = action[:, :, agent_idx, :]  # [1, 256, 17]
        td_logits = td["logits"].reshape(
            td["logits"].shape[0],
            td["logits"].shape[1],
            td["observation"].shape[-2],
            -1,
        )  # [1, 256, 2, 34]
        td_logit_agent_idx = td_logits[:, :, agent_idx, :]  # [1, 256, 34]
        loc, scale = self.dist_create(td_logit_agent_idx)  # [1, 256, 17]
        behaviour_action_log_probs = self.dist_log_prob(
            loc, scale, action_agent_idx
        )  # [1, 256]
        loc, scale = self.dist_create(policy_logits)  # [256, 17]
        target_action_log_probs = self.dist_log_prob(
            loc, scale, action_agent_idx
        )  # [1, 256]

        with torch.no_grad():
            vs, advantages = self.compute_gae(
                truncation=td["truncation"],
                termination=termination,
                reward=reward,
                values=baseline,
                bootstrap_value=bootstrap_value,
            )  # [3, 256]

        rho_s = torch.exp(
            target_action_log_probs - behaviour_action_log_probs
        )  # [1, 256]
        surrogate_loss1 = rho_s * advantages  # [3, 256]
        surrogate_loss2 = (
            rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
        )  # [3, 256]
        policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = vs - baseline  # [3, 256]
        v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5

        # Entropy reward
        entropy = torch.mean(self.dist_entropy(loc, scale))
        entropy_loss = self.entropy_cost * -entropy

        return policy_loss + v_loss + entropy_loss
