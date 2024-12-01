# The purpose of this file is to create a price tensor for input into the neural network
# and to train the policy using Deep Deterministic Policy Gradient.
# Code is inspired by the paper "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
# For more details, see: c.f https://arxiv.org/abs/1706.10059


from typing import Iterator, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from adaptivepm.models import Actor
from adaptivepm.portfolio import Portfolio, PortfolioVectorMemory

torch.set_default_device("mps")


class KrakenDataSet(Dataset):
    """Creates a tensor Xt as defined by equation 18 in paper to feed into ANN

    Parameters
    ----------
    Dataset : _type_
        Base class from torch utils class that allows to iterate over dataset
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
        return self.portfolio.nobs

    def __getitem__(self, idx):
        msecurities = self.portfolio.m_noncash_securities
        start = idx * self.step_size
        end = start + self.window_size
        xt = torch.zeros(3, msecurities, self.window_size).to(self.device)
        xt[0] = (self.close_pr[start:end:,] / self.close_pr[end - 1,]).T
        xt[1] = (self.high_pr[start:end:,] / self.close_pr[end - 1,]).T
        xt[2] = (self.low_pr[start:end:,] / self.close_pr[end - 1,]).T
        # updated_pvm = self.pvm[end - 2]
        return xt, end - 2


def custom_collate_fn(batch):
    """
    Custom collate function to batch data from KrakenDataSet without unnecessary copies.

    Args:
    - batch: List of tuples where each element is a (xt, pvm) pair.
      xt: A tensor of shape [3, msecurities, window_size].
      pvm: A scalar tensor.

    Returns:
    - A tuple (batched_xt, batched_pvm) where:
      - batched_xt: A tensor of shape [batch_size, 3, msecurities, window_size]
      - batched_pvm: A tensor of shape [batch_size]
    """
    # Separate the batch into xt (list of [3, msecurities, window_size] tensors)
    xt_batch = [item[0] for item in batch]  # List of xt tensors
    pvm_batch = [item[1] for item in batch]  # List of pvm tensors

    # Stack xt_batch along the batch dimension to create a single tensor of shape [batch_size, 3, msecurities, window_size]
    batched_xt = torch.stack(
        xt_batch, dim=0
    )  # [batch_size, 3, msecurities, window_size]

    # Stack pvm_batch along the batch dimension to create a single tensor of shape [batch_size]
    # [batch_size]

    return batched_xt, pvm_batch


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
        self.batch_size = batch_size  # This is now batch size instead of window size
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


def modify_tensor(my_tensor, idx):
    return my_tensor[idx]


def main():
    device = "mps"
    BATCH_SIZE = 50
    WINDOW_SIZE = 50
    STEP_SIZE = 1

    assets: List[str] = [
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

    port = Portfolio(asset_names=assets)

    # portfolio memory vector
    nsamples = port.nobs
    msecurities = port.m_noncash_securities
    # window_size - 1 pretains to index at time t
    # first index for pvm wt-1 is window_size - 2 index
    # pvm = (torch.ones(nobs, msecurities) / msecurities).to(device)
    pvm = PortfolioVectorMemory(nsamples=nsamples, msecurities=msecurities)
    kraken_ds = KrakenDataSet(port, window_size=WINDOW_SIZE)

    batch_sampler = SlidingWindowBatchSampler(
        kraken_ds, batch_size=BATCH_SIZE, step_size=STEP_SIZE
    )

    kraken_dl = DataLoader(
        kraken_ds,
        batch_size=1,
        batch_sampler=batch_sampler,
        # collate_fn=custom_collate_fn,
        pin_memory=True,
        generator=torch.Generator(device="mps"),
    )

    actor = Actor()

    for xt, wt_prev in kraken_dl:
        # returns the portfolio weight wt at time t which is the action
        wt = actor(xt, pvm.get_memory_stack(wt_prev))
        print(wt)


if __name__ == "__main__":
    # used for debugging purposes
    main()
