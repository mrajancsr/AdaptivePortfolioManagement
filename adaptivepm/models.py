# The purpose of this file is to create a deep deterministic policy gradient
# which is an ANN composed of actor and critic
# Actor will estimate the policy
# Critic will estimate the Q value function

import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_uniform(m.weight.data, nonlinearity="relu")
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Actor(nn.Module):
    def __init__(self, input_channels=3, output_dim=12):
        """_summary_

        Parameters
        ----------
        input_channels : int, optional
            number of features, default=3
        output_dim : int, optional
            number of asset weights, default=12
        """
        super(Actor, self).__init__()
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Conv2d(
                input_channels, 2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.ReLU(True),
            nn.Conv2d(2, 20, kernel_size=(1, 48), stride=1, padding=(0, 0)),
            nn.ReLU(True),
        )
        self.model2 = nn.Sequential(nn.Conv2d(21, 1, kernel_size=(1, 1), stride=1))
        self.cash_bias = nn.Parameter(torch.full((1, 1, 1), 0.2))
        self.softmax = nn.Softmax(dim=2)
        self.apply(weights_init)

    def forward(self, input: torch.tensor, pvm: torch.tensor) -> torch.tensor:
        """performs a forward pass and returns portfolio weights at time t
        c.f pg 14-15 from https://arxiv.org/pdf/1706.10059

        Parameters
        ----------
        input : torch.tensor
            price tensor Xt comprised of close, high and low prices
            dim = (batch_size, features, msecurities, window_size)
            window_size pretains to last 50 trading prices
        pvm : torch.tensor
            weights w(t-1) from portfolio vector memory at time t-1
            dim = (batch_size, mfeatures)

        Returns
        -------
        torch.tensor, dim = (batch_size, mfeatures)
            action at time t
        """
        batch_size = input.shape[0]
        x = self.model(input)
        prev_weights = pvm.unsqueeze(2)
        prev_weights = prev_weights.repeat(1, 1, 1)
        x = torch.cat([x, prev_weights.unsqueeze(1)], dim=1)
        x = self.model2(x)
        cash_bias = self.cash_bias.expand(batch_size, 1, 1, 1)
        x = torch.cat([cash_bias, x], dim=2)
        portfolio_weights = self.softmax(x)

        return portfolio_weights.view(batch_size, self.output_dim)
