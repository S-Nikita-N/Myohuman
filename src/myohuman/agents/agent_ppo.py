import math
import time
import torch
import logging
import numpy as np

from typing import List, Optional, Tuple

from myohuman.agents.agent import Agent
from myohuman.learning.learning_utils import to_test, to_train, estimate_advantages


logger = logging.getLogger(__name__)


class AgentPPO(Agent):

    def __init__(
        self,
        tau: float = 0.95,
        clip_epsilon: float = 0.2,
        mini_batch_size: int = 64,
        opt_num_epochs: int = 1,
        value_opt_niter: int = 1,
        use_mini_batch: bool = False,
        policy_grad_clip: Optional[List[Tuple[torch.nn.Module, float]]] = None,
        optimizer_policy: Optional[torch.optim.Optimizer] = None,
        optimizer_value: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ):
        """
        Initialize the PPO Agent.

        Args:
            clip_epsilon (float): Clipping parameter for PPO's surrogate objective.
            mini_batch_size (int): Size of mini-batches for stochastic gradient descent.
            use_mini_batch (bool): Whether to use mini-batch updates.
            policy_grad_clip (List[Tuple[torch.nn.Module, float]], optional):
                List of tuples containing networks and their max gradient norms for clipping.
            tau (float): GAE parameter for bias-variance trade-off.
            optimizer_policy (torch.optim.Optimizer, optional): Optimizer for the policy network.
            optimizer_value (torch.optim.Optimizer, optional): Optimizer for the value network.
            opt_num_epochs (int): Number of epochs for policy updates.
            value_opt_niter (int): Number of iterations for value network updates.
            **kwargs: Additional parameters for the base Agent class.
        """
        super().__init__(**kwargs)
       
        self.tau = tau
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.opt_num_epochs = opt_num_epochs
        self.value_opt_niter = value_opt_niter
        # Initialize PPO parameters
        self.clip_epsilon = clip_epsilon
        self.mini_batch_size = mini_batch_size
        self.use_mini_batch = use_mini_batch
        self.policy_grad_clip = policy_grad_clip

    def update_params(self, batch) -> float:
        """
        Perform parameter updates for both policy and value networks using the collected batch.

        Args:
            batch: A batch of collected experiences containing states, actions, rewards, masks, and exploration flags.

        Returns:
            float: Time taken to perform the parameter updates.
        """
        t0 = time.time()
        # Set the modules to training mode
        to_train(self.policy_net, self.value_net)

        # Convert the batch to tensors
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)

        # Compute value estimates for the states without gradient tracking
        with to_test(self.policy_net, self.value_net):
            with torch.no_grad():
                values = self.value_net(states)

        # Estimate advantages and returns
        advantages, returns = estimate_advantages(
            rewards, masks, values, self.gamma, self.tau
        )

        self.update_policy(states, actions, returns, advantages, exps)

        return time.time() - t0

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        exps: torch.Tensor,
    ) -> None:
        """
        Update the policy network using PPO's clipped surrogate objective.

        Args:
            states (torch.Tensor): Tensor of states.
            actions (torch.Tensor): Tensor of actions taken.
            returns (torch.Tensor): Tensor of target returns.
            advantages (torch.Tensor): Tensor of advantage estimates.
            exps (torch.Tensor): Tensor indicating exploration flags.
        """
        # Compute log proabilities of the actions under the current policy
        with to_test(self.policy_net, self.value_net):
            with torch.no_grad():
                fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        for _ in range(self.opt_num_epochs):
            if self.use_mini_batch:
                perm = np.arange(states.shape[0])
                np.random.shuffle(perm)
                perm = torch.LongTensor(perm).to(self.device)

                states, actions, returns, advantages, fixed_log_probs, exps = (
                    states[perm].clone(),
                    actions[perm].clone(),
                    returns[perm].clone(),
                    advantages[perm].clone(),
                    fixed_log_probs[perm].clone(),
                    exps[perm].clone(),
                )

                optim_iter_num = int(math.floor(states.shape[0] / self.mini_batch_size))
                for i in range(optim_iter_num):
                    ind = slice(
                        i * self.mini_batch_size,
                        min((i + 1) * self.mini_batch_size, states.shape[0]),
                    )
                    (
                        states_b,
                        actions_b,
                        advantages_b,
                        returns_b,
                        fixed_log_probs_b,
                        exps_b,
                    ) = (
                        states[ind],
                        actions[ind],
                        advantages[ind],
                        returns[ind],
                        fixed_log_probs[ind],
                        exps[ind],
                    )
                    ind = exps_b.nonzero(as_tuple=False).squeeze(1)
                    self.update_value(states_b, returns_b)
                    surr_loss = self.ppo_loss(
                        states_b, actions_b, advantages_b, fixed_log_probs_b, ind
                    )
                    self.optimizer_policy.zero_grad()
                    surr_loss.backward()
                    self.clip_policy_grad()
                    self.optimizer_policy.step()
            else:
                ind = exps.nonzero(as_tuple=False).squeeze(1)
                self.update_value(states, returns)
                surr_loss = self.ppo_loss(
                    states, actions, advantages, fixed_log_probs, ind
                )
                self.optimizer_policy.zero_grad()
                surr_loss.backward()
                self.clip_policy_grad()
                self.optimizer_policy.step()

    def update_value(self, states: torch.Tensor, returns: torch.Tensor) -> None:
        """
        Update the critic (value network) by minimizing the MSE between predicted values and returns.

        Args:
            states (torch.Tensor): Tensor of states.
            returns (torch.Tensor): Tensor of target returns.
        """
        for _ in range(self.value_opt_niter):
            values_pred = self.value_net(states)
            value_loss = (values_pred - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def clip_policy_grad(self) -> None:
        """
        Clip gradients of the policy network to prevent exploding gradients.
        """
        if self.policy_grad_clip is not None:
            for net, max_norm in self.policy_grad_clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)

    def ppo_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        fixed_log_probs: torch.Tensor,
        ind: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the PPO surrogate loss.

        Args:
            states (torch.Tensor): Tensor of states.
            actions (torch.Tensor): Tensor of actions taken.
            advantages (torch.Tensor): Tensor of advantage estimates.
            fixed_log_probs (torch.Tensor): Tensor of log probabilities under the old policy.
            ind (torch.Tensor): Tensor of indices indicating active exploration flags.

        Returns:
            torch.Tensor: Computed PPO surrogate loss.
        """
        log_probs = self.policy_net.get_log_prob(states[ind], actions[ind])
        ratio = torch.exp(log_probs - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages
        )
        surr_loss = -torch.min(surr1, surr2).mean()

        return surr_loss
