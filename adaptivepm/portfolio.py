from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterator, List

import pandas as pd

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
    m_assets: int = field(init=False, default=0)
    m_noncash_assets: int = field(init=False, default=0)

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
        self.m_assets = len(self.__assets)
        self.m_noncash_assets = self.m_assets - 1
        self.n_samples = self.__prices["close"].shape[0]

    def _load_pickle_object(self):
        with open(PATH_TO_PRICES_PICKLE, "rb") as f:
            self.__prices.update(pickle.load(f))

    def __iter__(self) -> Iterator[Asset]:
        yield from self.assets()

    def __repr__(self) -> str:
        return f"Portfolio size: {self.m_assets} \
            \nm_assets: {[asset.name for asset in self.assets()]}"

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


if __name__ == "__main__":
    # used for debugging purposes
    m_assets: List[str] = [
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
    port = Portfolio(asset_names=m_assets)
    print(port)
