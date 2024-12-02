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
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(2, 20, kernel_size=(1, 48), stride=1, padding=(0, 0)),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.model2 = nn.Sequential(nn.Conv2d(21, 1, kernel_size=(1, 1), stride=1))
        self.cash_bias = nn.Parameter(torch.full((1, 1, 1), 0.2))
        self.softmax = nn.Softmax(dim=2)
        self.apply(weights_init)

    def forward(self, price_tensor: torch.tensor, pvm: torch.tensor) -> torch.tensor:
        """performs a forward pass and returns portfolio weights at time t
        c.f pg 14-15 from https://arxiv.org/pdf/1706.10059

        Parameters
        ----------
        price_tensor : torch.tensor
            price tensor Xt comprised of close, high and low prices
            dim = (batch_size, kfeatures, massets, window_size)
            window_size pretains to last 50 trading prices
        pvm : torch.tensor
            weights w(t-1) from portfolio vector memory at time t-1
            dim = (batch_size, massets)

        Returns
        -------
        torch.tensor, dim = (batch_size, massets)
            action at time t
        """
        batch_size = price_tensor.shape[0]
        x = self.model(price_tensor)
        prev_weights = pvm.unsqueeze(2).repeat(1, 1, 1).unsqueeze(1)
        x = torch.cat([x, prev_weights], dim=1)
        x = self.model2(x)
        cash_bias = self.cash_bias.expand(batch_size, 1, 1, 1)
        x = torch.cat([cash_bias, x], dim=2)
        portfolio_weights = self.softmax(x)

        return portfolio_weights.view(batch_size, self.output_dim)


class Critic(nn.Module):
    def __init__(self, input_channels: int = 3, m_assets: int = 11):
        """Critic network for DDPG.  Given a state (Xt, w(t-1)), this network outputs the Q-Value

        Parameters
        ----------
        input_channels : int
            _description_
        output_dim : int
            _description_
        """
        super(Critic, self).__init__()
        self.m_assets = m_assets
        self.model = nn.Sequential(
            nn.Conv2d(
                input_channels, 2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(2, 20, kernel_size=(1, 48), stride=1, padding=(0, 0)),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(21, 20, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Flatten(),
            nn.Linear(20 * m_assets, 1),
        )
        self.apply(weights_init)

    def forward(self, price_tensor: torch.tensor, pvm: torch.tensor) -> torch.tensor:
        """performs forward pass and returns Q value for each price-action pair

        Parameters
        ----------
        price_tensor : torch.tensor
            price tensor Xt comprised of close, high and low prices
            dim = (batch_size, kfeatures, m_assets, window_size)
            window_size pretains to last 50 trading prices
        pvm : torch.tensor
            the previous weights w(t-1) which is the previous action

        Returns
        -------
        torch.tensor
            Q value for each state-action pair
        """
        x = self.model(price_tensor)
        prev_weights = pvm.unsqueeze(2).repeat(1, 1, 1).unsqueeze(1)
        x = torch.cat([x, prev_weights], dim=1)
        # estimate the q value
        q_value = self.model2(x)
        return q_value
