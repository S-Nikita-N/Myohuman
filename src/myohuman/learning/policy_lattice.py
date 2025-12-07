import torch.nn as nn
from myohuman.learning.policy import Policy
from myohuman.learning.mlp import MLP
from myohuman.learning.running_norm import RunningNorm
import torch

from torch.distributions import MultivariateNormal


class PolicyLattice(Policy):
    def __init__(self, cfg, action_dim, latent_dim, state_dim, net_out_dim=None):
        super().__init__()
        self.type = "lattice"
        self.norm = RunningNorm(state_dim)
        self.net = net = MLP(
            state_dim,
            cfg.learning.mlp.units,
            cfg.learning.mlp.activation
        )

        self.action_dim = action_dim
        self.latent_dim = latent_dim

        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.action_mean = nn.Linear(net_out_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.log_std = nn.Parameter(
            torch.ones(1, action_dim + latent_dim) * cfg.learning.log_std,
            requires_grad=(not cfg.learning.fix_std)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.net(x)
        action_mean = self.action_mean(x)
        std = torch.exp(self.log_std)
        action_var = std[:, :self.action_dim] ** 2
        latent_var = std[:, self.action_dim:] ** 2

        sigma_mat = (self.action_mean.weight * latent_var[..., None, :]).matmul(self.action_mean.weight.T)
        sigma_mat[..., torch.arange(self.action_dim), torch.arange(self.action_dim)] += action_var
        self.lattice_dist = MultivariateNormal(action_mean, sigma_mat)
        return self.lattice_dist

    # def forward(self, x):
    #     B = x.size(0)
    #     x = self.norm(x)
    #     x = self.net(x)
    #     action_mean = self.action_mean(x)  # [B, A]

    #     std = torch.exp(self.log_std)
    #     action_std = std[:, :self.action_dim]
    #     latent_std = std[:, self.action_dim:]
        
    #     action_var = action_std.pow(2).flatten()   # [A]
    #     latent_var = latent_std.pow(2).flatten()   # [L]
    #     W = self.action_mean.weight                # [A, H]

    #     latent_cov = W @ torch.diag(latent_var) @ W.T  # [A, A]
    #     Sigma = latent_cov + torch.diag(action_var)    # [A, A]
    #     Sigma_batch = Sigma.expand(B, -1, -1)          # [B, A, A]

    #     dist = MultivariateNormal(action_mean, Sigma_batch)
    #     return dist

    def select_action(self, x, mean_action=False):
        dist = self.forward(x) 
        action = dist.loc if mean_action else dist.rsample()
        return action
    
    def get_log_prob(self, x, value):
        dist = self.forward(x)
        return dist.log_prob(value).unsqueeze(1)
