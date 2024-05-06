from typing import Dict, Sequence, Tuple, ClassVar
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F

from Humanoid_MARL.models.mlp import MLP


class Agent(nn.Module):
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
        super(Agent, self).__init__()

        if network_config.get("LSTM"):
            raise NotImplementedError("LSTM not supported yet")
        elif network_config.get("TRANSFORMER"):
            raise NotImplementedError("Transformer not supported yet")
        else:
            self.policy = MLP.construct(policy_layers)
            self.value = MLP.construct(value_layers)

        self.num_steps = torch.zeros((), device=device)
        self.running_mean = torch.zeros(policy_layers[0], device=device)
        self.running_variance = torch.zeros(policy_layers[0], device=device)
        self.loss_dict = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
        }

        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = 0.95
        self.epsilon = 0.2
        self.device = device
        self.min_entropy_loss = 0.0
        self.max_entropy_loss = 100.0  # TODO : Verify that this is true

    def _reset_loss(self):
        self.loss_dict = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
        }

    @torch.jit.export
    def dist_create(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normal followed by tanh.
        torch.distribution doesn't work with torch.jit, so we roll our own.

        Parameters: logits: torch.Tensor : Logits from the policy network that vary the mean and scale of the normal distribution

        Returns: Tuple[torch.Tensor, torch.Tensor] : Tuple of loc and scale of the normal distribution
        """
        loc, scale = torch.split(
            logits, logits.shape[-1] // 2, dim=-1
        )  # split the logits in half into mean and std
        # softplus(x) = log(1 + exp(x)) : This is used to ensure that the scale is always positive where x = 0 -> log(1 + exp(0)) = log(1 + 1) = log(2) = 0.6931
        scale = F.softplus(scale) + 0.001
        return loc, scale

    @torch.jit.export
    def dist_sample_no_postprocess(
        self, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample from a normal distribution with mean loc and scale scale.
        This is a n-dimensional normal distribution where n is the size of the loc and scale tensors.

        Parameters: loc: torch.Tensor : Mean of the normal distribution

        Parameters: scale: torch.Tensor : Scale of the normal distribution
        """
        if (
            torch.any(scale <= 0)
            or torch.isnan(scale).any()
            or torch.isinf(scale).any()
        ):
            logging.error("Scale is negative, nan or inf : ", scale)
            scale = torch.abs(scale) + 1e-6
        return torch.normal(loc, scale)

    @classmethod
    def dist_postprocess(cls, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @torch.jit.export
    def dist_entropy(self, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Calculate the entropy of a normal distribution given mean and scale parameters.
        Entropy Formula: H(P) = E_x ~ P[-log(P(x))]

        Parameters:
            loc (torch.Tensor): Mean parameter of the normal distribution.
            scale (torch.Tensor): Standard deviation parameter of the normal distribution.

        Returns:
            torch.Tensor: Entropy of the normal distribution.
        """
        if (
            torch.any(scale <= 0)
            or torch.isnan(scale).any()
            or torch.isinf(scale).any()
        ):
            logging.error("Scale is negative, nan or inf : ", scale)
            scale = torch.abs(scale) + 1e-6
        # Step 1: Calculate the log normalization term
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(
            scale
        )  # entropy formula: 0.5 * log(2 * pi * e) + log(std)
        # Step 2: Compute entropy approximation
        entropy = 0.5 + log_normalized
        # Step 3: Expand entropy to match the shape of loc
        entropy = entropy * torch.ones_like(loc)
        # Step 4: Sample from normal distribution
        dist = torch.normal(loc, scale)
        # Step 5: Compute log determinant of Jacobian
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        # Step 6: Add log determinant of Jacobian to entropy
        entropy = entropy + log_det_jacobian
        # Step 7: Sum entropy across dimensions
        return entropy.sum(dim=-1)

    @torch.jit.export
    def dist_log_prob(
        self, loc: torch.Tensor, scale: torch.Tensor, dist: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the log probability of a distribution given mean and scale parameters.

        Parameters:
            loc (torch.Tensor): Mean parameter of the normal distribution.
            scale (torch.Tensor): Standard deviation parameter of the normal distribution.
            dist (torch.Tensor): Input distribution to compute the log probability for.

        Returns:
            torch.Tensor: Log probability of the input distribution.
        """
        # Step 1: Calculate the unnormalized log probability
        log_unnormalized = -0.5 * ((dist - loc) / scale).square()

        # Step 2: Calculate the log normalization term
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)

        # Step 3: Calculate the log determinant of the Jacobian
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))

        # Step 4: Calculate the overall log probability
        log_prob = log_unnormalized - log_normalized - log_det_jacobian

        # Step 5: Sum the log probabilities across dimensions
        return log_prob.sum(dim=-1)

    @torch.jit.export
    def update_normalization(self, observation: torch.Tensor) -> None:
        """
        Update the running mean and running variance used for normalization.

        Parameters:
            observation (torch.Tensor): Batch of observations to update normalization statistics.

        Returns:
            None
        """
        # Update the total number of steps
        self.num_steps += observation.shape[0] * observation.shape[1]
        # Calculate the difference between the current observation and the running mean
        input_to_old_mean = observation - self.running_mean
        # Compute the mean difference across all observations in the batch, rollout dim, and time steps
        mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
        # Update the running mean by adding the mean difference
        self.running_mean = self.running_mean + mean_diff
        # Calculate the difference between the current observation and the updated running mean
        input_to_new_mean = observation - self.running_mean
        # Compute the variance difference by element-wise multiplication
        var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
        # Update the running variance by adding the variance difference
        self.running_variance = self.running_variance + var_diff

    @torch.jit.export
    def normalize(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Given the running variance of size [obs_dim], normalize the observation among the rollout and batch dim.
        This function helps stabilize the training process and helps in faster convergence by normalizing
        the potential gradients from the observation (a parameter to actor and critic).

        Parameters: observation: torch.Tensor : Observation to be normalized

        Returns: torch.Tensor : Normalized observation

        """
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = torch.clip(variance, 1e-6, 1e6)
        # Converting to z-score space and clipping to [-5, 5]
        return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

    @torch.jit.export
    def get_logits_action(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get logits and sample action from the policy network.

        Parameters:
            observation (torch.Tensor): Input observation to the policy network.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing logits and sampled action.
        """
        observation = self.normalize(observation)
        logits = self.policy(observation)
        # creates the raw parameters for the normal distribution: mean and std
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
        """
        Compute Generalized Advantage Estimation (GAE) for advantages.

        Parameters:
            truncation (torch.Tensor): Truncation flag indicating end of trajectory.
            termination (torch.Tensor): Termination flag indicating end of episode.
            reward (torch.Tensor): Rewards received at each time step.
            values (torch.Tensor): Estimated value function at each time step.
            bootstrap_value (torch.Tensor): Bootstrapped value at the end of trajectory.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing value estimates and advantages.
        """
        # Step 1: Compute the truncation mask
        truncation_mask = 1 - truncation

        # Step 2: Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )

        # Step 3: Compute temporal differences (deltas)
        deltas = (
            reward + self.discounting * (1 - termination) * values_t_plus_1 - values
        )
        # Multiply deltas by the truncation mask to handle truncated trajectories
        deltas *= truncation_mask

        # Step 4: Compute accumulated advantages using backward recursion
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

        # Step 5: Add V(x_s) to get v_s
        vs = vs_minus_v_xs + values
        # Compute Vs(t+1)
        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)

        # Step 6: Compute advantages
        advantages = (
            reward + self.discounting * (1 - termination) * vs_t_plus_1 - values
        ) * truncation_mask

        return vs, advantages

    @torch.jit.export
    def loss(self, td: Dict[str, torch.Tensor], agent_idx: int):
        """
        Compute the loss function for the Proximal Policy Optimization (PPO) algorithm.

        Parameters:
            td (Dict[str, torch.Tensor]): Training data containing observation, action, reward, done, truncation, logits.
            agent_idx (int): Index of the agent.

        Returns:
            torch.Tensor: Total loss computed as the sum of policy loss, value loss, and entropy loss.
        """
        self._reset_loss()  # reset the loss values in the loss dictionary
        # recieve the observation, action, reward, done, truncation, logits from the training data based on the current agent
        td_obs = td["observation"][
            :, :, agent_idx, :
        ]  # [unroll_length, batch_size, obs_dim]
        # Normalize the observation
        observation = self.normalize(td_obs)  # [unroll_length, batch_size, obs_dim]
        # obtain the logits for the policy and the baseline value for the observation [s1, ..., s_t] where t = unroll_length
        policy_logits = self.policy(
            observation[:-1]
        )  # [unroll_length, batch_size, action_dim] : [2, 256, 34]
        # obtain the baseline value for the observation [s1, ..., s_t] where t = unroll_length
        baseline = self.value(observation)
        baseline = torch.squeeze(
            baseline, dim=-1
        )  # [unroll_length, batch_size]  [3, 256]

        # Use last baseline value (from the value function) to bootstrap.
        # The value function is trained to predict the sum of rewards from here on.
        bootstrap_value = baseline[-1]  # [256]
        # Remove the last baseline value from the baseline tensor so we have [s1, ..., s_t] where t = unroll_length-1
        baseline = baseline[:-1]  # [unroll_length-1, batch_size] [2, 256]

        # Get the reward, action, logits for the agent at agent_idx
        reward = td["reward"][:, :, agent_idx] * self.reward_scaling
        # Get the done and truncation values for the agent at agent_idx
        termination = td["done"] * (1 - td["truncation"])
        # Get the action for the agent at agent_idx [rollout_length, batch_size, action_dim]
        action = td["action"].reshape(
            td["action"].shape[0],
            td["action"].shape[1],
            td["observation"].shape[-2],
            -1,
        )  # [unroll_length, batch_size, num_agents, action_dim]
        action_agent_idx = action[
            :, :, agent_idx, :
        ]  # [unroll_length, batch_size, action_dim] : [2, 246, 17]
        td_logits = td["logits"].reshape(
            td["logits"].shape[0],
            td["logits"].shape[1],
            td["observation"].shape[-2],
            -1,
        )  # [unroll_length, batch_size, num_agents, action_dim] : [2, 256, 2, 34]
        # Get the logits from a previous trajectort in the PPO step
        td_logit_agent_idx = td_logits[
            :, :, agent_idx, :
        ]  # [unroll_length, batch_size, action_dim] : [2, 256, 34]

        loc, scale = self.dist_create(
            td_logit_agent_idx
        )  # p(a | s)_old [unroll_length-1, batch_size, action_dim], [unroll_length-1, batch_size, action_dim]
        behaviour_action_log_probs = self.dist_log_prob(
            loc, scale, action_agent_idx
        )  # [unroll_length-1, batch_size]
        loc, scale = self.dist_create(
            policy_logits
        )  # [unroll_length-1, batch_size, action_dim], [unroll_length-1, batch_size, action_dim
        target_action_log_probs = self.dist_log_prob(
            loc, scale, action_agent_idx
        )  # p(a | s)_new [unroll_length-1, batch_size]

        with torch.no_grad():
            vs, advantages = self.compute_gae(
                truncation=td["truncation"],
                termination=termination,
                reward=reward,
                values=baseline,
                bootstrap_value=bootstrap_value,
            )

        rho_s = torch.exp(
            target_action_log_probs
            - behaviour_action_log_probs  # log(p(a | s)_new) - log(p(a | s)_old) = log(p(a | s)_new / p(a | s)_old) -> ratio between the new and old policy
        )  # [unroll_length-1, batch_size]
        # print("rho_s: ", rho_s)
        surrogate_loss1 = rho_s * advantages  # rho_s * A(s, a)
        surrogate_loss2 = (
            rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
        )  # PPO clip loss objectvive
        policy_loss = -torch.mean(
            torch.minimum(surrogate_loss1, surrogate_loss2)
        )  # -E[min(rho_s * A(s, a), rho_s.clip(1 - epsilon, 1 + epsilon) * A(s, a))] -> We want to maximize the expected advantage of the policy
        # print("policy_loss: ", policy_loss.item())

        # Value function loss
        v_error = vs - baseline  # value expectation - value prediction from network
        v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5  # L2 loss
        # print("v_loss: ", v_loss.item())

        # Entropy reward
        is_negative = torch.any(scale <= 0)
        if (
            not is_negative
            and not torch.isnan(scale).any()
            and not torch.isinf(scale).any()
        ):
            scale = torch.abs(scale) + 1e-6
            entropy = torch.mean(self.dist_entropy(loc, scale))
            entropy_loss = self.entropy_cost * -entropy
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)
        # entropy_loss = (self.entropy_cost * -entropy).clamp(
        #     self.min_entropy_loss, self.max_entropy_loss
        # )  # multiply the entropy coffeicent by the entropy cost to encourage exploration
        # print("entropy_loss: ", entropy_loss.item())

        self.loss_dict["policy_loss"] = policy_loss.item()
        self.loss_dict["value_loss"] = v_loss.item()
        self.loss_dict["entropy_loss"] = entropy_loss.item()
        self.loss_dict["total_loss"] = (
            policy_loss.item() + v_loss.item() + entropy_loss.item()
        )
        return policy_loss + v_loss + entropy_loss
