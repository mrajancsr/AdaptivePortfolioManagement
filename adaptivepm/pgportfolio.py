from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from adaptivepm import Asset

PATH_TO_PRICES_PICKLE = os.path.join(
    os.getcwd(), "datasets", "Kraken_pipeline_output", "prices.pkl"
)


@dataclass
class Portfolio:
    """Implements a Portfolio that holds CryptoCurrencies as Assets
    Parameters
    _ _ _ _ _ _ _ _ _ _


    Attributes:
    _ _ _ _ _ _ __ __ _ _
    __assets: Dict[name, Asset]
        dictionary of Asset objects whose keys are the asset names
    """

    asset_names: List[str]
    __prices: Dict[str, pd.DataFrame] = field(init=False, default_factory=lambda: {})
    __assets: Dict[str, Asset] = field(init=False)
    nsecurities: int = field(init=False, default=0)

    def __post_init__(self):
        self._load_pickle_object()
        self.__assets = {
            asset_name: Asset(
                name=asset_name,
                open_price=self.__prices["open"][asset_name],
                close_price=self.__prices["close"][asset_name],
                high_price=self.__prices["high"][asset_name],
                low_price=self.__prices["low"][asset_name],
            )
            for asset_name in self.asset_names
        }
        self.nsecurities = len(self.__assets)
        self.nobs = self.__prices["close"].shape[0]

    def _load_pickle_object(self):
        with open(PATH_TO_PRICES_PICKLE, "rb") as f:
            self.__prices.update(pickle.load(f))

    def __iter__(self) -> Iterator[Asset]:
        yield from self.assets()

    def __repr__(self) -> str:
        return f"Portfolio size: {self.security_count} \
            \nAssets: {[asset.name for asset in self.assets()]}"

    def get_asset(self, name: str) -> Asset:
        """Returns the asset in the portfolio given the name of the asset

        Parameters
        ----------
        asset : str
            name of the asset

        Returns
        -------
        Asset
            contains information about the asset
        """
        return self.__assets.get(name.upper())

    def assets(self) -> Iterator[Asset]:
        yield from self.__assets.values()

    def get_relative_price(self):
        return self.__prices["close_to_open"]

    def get_close_price(self):
        return self.__prices["close"]

    def get_high_price(self):
        return self.__prices["high"]

    def get_low_price(self):
        return self.__prices["low"]

    def generate_portfolio_weights(self):
        num_periods = next(self).size
        total_assets = self.security_count
        weights = np.zeros((num_periods, total_assets))
        weights[0, 0] = 1.0
        return weights


class KrakenDataSet(Dataset):
    def __init__(self, portfolio: Portfolio, nperiods: int = 50):
        self.portfolio = portfolio
        self.nperiods = nperiods
        self.close_pr = torch.tensor(self.portfolio.get_close_price().values)
        self.high_pr = torch.tensor(self.portfolio.get_high_price().values)
        self.low_pr = torch.tensor(self.portfolio.get_low_price().values)

    def __len__(self):
        return self.portfolio.total_observations

    def __getitem__(self, idx):
        nsecurities = self.portfolio.nsecurities
        xt = torch.zeros(3, self.nperiods, nsecurities)
        xt[0] = (
            self.close_pr[idx : self.nperiods + idx :,]
            / self.close_pr[self.nperiods + idx - 1,]
        )
        xt[1] = (
            self.high_pr[idx : self.nperiods + idx :,]
            / self.close_pr[self.nperiods + idx - 1,]
        )
        xt[2] = (
            self.low_pr[idx : self.nperiods + idx :,]
            / self.close_pr[self.nperiods + idx - 1,]
        )
        return xt


def main():
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
    kraken_ds = KrakenDataSet(port)
    for xt in kraken_ds:
        print(xt)


main()
