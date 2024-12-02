# The purpose of this file is to create a price tensor for input into the neural network
# and to train the policy using Deep Deterministic Policy Gradient.
# Code is inspired by the paper "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
# For more details, see: c.f https://arxiv.org/abs/1706.10059


from typing import Iterator, List

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from adaptivepm.models import Actor
from adaptivepm.portfolio import Portfolio, PortfolioVectorMemory

torch.set_default_device("mps")


class KrakenDataSet(Dataset):
    """Creates a tensor Xt as defined by equation 18 in paper to feed into ANN

    Parameters
    ----------
    Dataset : torch.utils.data.dataset.Dataset
        ABC from torch utils class that represents a dataset
    """

    def __init__(
        self,
        portfolio: Portfolio,
        window_size: int = 50,
        step_size: int = 1,
        device="mps",
    ):
        self.portfolio = portfolio
        self.window_size = window_size
        self.step_size = step_size
        self.close_pr = torch.tensor(
            self.portfolio.get_close_price().values[:, 1:], dtype=torch.float32
        ).to(device)
        self.high_pr = torch.tensor(
            self.portfolio.get_high_price().values[:, 1:], dtype=torch.float32
        ).to(device)
        self.low_pr = torch.tensor(
            self.portfolio.get_low_price().values[:, 1:], dtype=torch.float32
        ).to(device)
        self.device = device

    def __len__(self):
        return self.portfolio.n_samples

    def __getitem__(self, idx):
        m_noncash_assets = self.portfolio.m_noncash_assets
        start = idx * self.step_size
        end = start + self.window_size

        # the price tensor
        xt = torch.zeros(3, m_noncash_assets, self.window_size).to(self.device)
        xt[0] = (self.close_pr[start:end:,] / self.close_pr[end - 1,]).T
        xt[1] = (self.high_pr[start:end:,] / self.close_pr[end - 1,]).T
        xt[2] = (self.low_pr[start:end:,] / self.close_pr[end - 1,]).T
        return xt, end - 2


class SlidingWindowBatchSampler(Sampler):
    """
    A custom BatchSampler that samples batches of size `batch_size` from KrakenDataSet
    using a sliding window approach.
    """

    def __init__(
        self, dataset: KrakenDataSet, batch_size: int = 50, step_size: int = 1
    ):
        """
        Parameters
        ----------
        dataset : KrakenDataSet
            The KrakenDataSet Object
        batch_size : int, optional
            size of the batch, default=50
        step_size : int, optional
            the step size for sliding window, default=1
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.step_size = step_size

        # we are getting data from dataset, so first index starts at index 0
        self.start_index = 0
        # The end index is dataset size
        self.end_index = len(dataset)  # the number of observations in your dataset

    def __iter__(self) -> Iterator[int]:
        """
        Yield batches of indices to sample from the dataset.
        Each batch contains indices from [i, i+batch_size).
        """
        # Start iterating from the start index (49)
        for i in range(
            self.start_index, self.end_index - self.batch_size + 1, self.step_size
        ):
            # Each batch will be a list of indices [i, i+batch_size)
            yield list(range(i, i + self.batch_size))

    def __len__(self) -> int:
        """Returns the number of batches."""
        return (self.end_index - self.batch_size) // self.step_size + 1


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
    pvm = PortfolioVectorMemory(n_samples=n_samples, m_assets=m_noncash_assets)

    actor = Actor()

    for xt, prev_index in kraken_dl:
        # returns the portfolio weight wt at time t which is the action
        wt = actor(xt, pvm.get_memory_stack(prev_index))

        # update the pvm with non cash assets only
        pvm.update_memory_stack(wt[:, 1:], prev_index + 1)


if __name__ == "__main__":
    # used for debugging purposes
    main()
