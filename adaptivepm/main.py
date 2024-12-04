# The purpose of this file is to create a price tensor for input into the neural network
# and to train the policy using Deep Deterministic Policy Gradient.
# Code is inspired by the paper "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
# For more details, see: c.f https://arxiv.org/abs/1706.10059


from typing import List

import torch
from torch.utils.data import DataLoader

from adaptivepm.dataset import (
    KrakenDataSet,
    SlidingWindowBatchSampler,
    get_current_and_next_batch,
)
from adaptivepm.memory import PortfolioVectorMemory
from adaptivepm.models import Actor, Critic
from adaptivepm.portfolio import Portfolio

torch.set_default_device("mps")


def main():
    BATCH_SIZE = 50  # training is done in mini-batches
    WINDOW_SIZE = 50  # last n trading days for the price tensor
    STEP_SIZE = 1  # for rolling window batch sampler
    DEVICE = "mps"

    asset_names: List[str] = [
        "CASH",
        "SOL",
        "ADA",
        "USDT",
        "AVAX",
        "LINK",
        "DOT",
        "PEPE",
        "ETH",
        "XRP",
        "TRX",
        "MATIC",
    ]

    port = Portfolio(asset_names=asset_names)

    # create the dataset which returns price tensor and t-2 index
    kraken_ds = KrakenDataSet(port, window_size=WINDOW_SIZE)

    # a rolling batch is needed for financial time series
    batch_sampler = SlidingWindowBatchSampler(
        kraken_ds, batch_size=BATCH_SIZE, step_size=STEP_SIZE
    )

    kraken_dl = DataLoader(
        kraken_ds,
        batch_size=1,
        batch_sampler=batch_sampler,
        pin_memory=True,
        generator=torch.Generator(device=DEVICE),
    )

    # portfolio memory vector to keep track of previous weights wt_prev
    n_samples = port.n_samples
    m_noncash_assets = port.m_noncash_assets

    # window_size - 1 is index at time t
    # window_size - 2 is index at time t-1
    # Hence first index for pvm wt-1 is window_size - 2 index
    pvm = PortfolioVectorMemory(n_samples=n_samples, m_noncash_assets=m_noncash_assets)

    actor = Actor()
    critic = Critic()
    yt = torch.randint(0, 10, (50, 11))
    for xt, xt_next, prev_index in get_current_and_next_batch(kraken_dl):
        # get previous weight w(t-1)
        wt_prev = pvm.get_memory_stack(prev_index)
        st = (xt, wt_prev)
        # returns the portfolio weight wt at time t which is the action
        wt = actor(*st)

        # update the pvm with non cash assets only
        pvm.update_memory_stack(wt, prev_index + 1)

        st_next = (xt_next, wt)
        ut = port.get_transacton_remainder_factor(wt, yt, wt_prev, n_iter=3)
        q_value = critic(*st)


if __name__ == "__main__":
    # used for debugging purposes
    main()
