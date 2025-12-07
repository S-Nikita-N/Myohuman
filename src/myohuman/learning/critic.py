import torch.nn as nn

from myohuman.learning.mlp import MLP


class Value(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=(128, 128),
        activation="tanh"
    ):
        """
        Simple value network: MLP feature extractor followed by linear value head.
        """
        super().__init__()
        self.net = MLP(input_dim, hidden_dims, activation)
        self.value_head = nn.Linear(self.net.out_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.net(x)
        value = self.value_head(x)
        return value
