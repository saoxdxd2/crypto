from __future__ import annotations

from freqtrade.strategy import IStrategy
from pandas import DataFrame


class RandomEntryBaseline(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 10
    minimal_roi = {"0": 0.015, "180": 0}
    stoploss = -0.025
    trailing_stop = False
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        date_ns = dataframe["date"].astype("int64")
        dataframe["random_bucket"] = ((date_ns // 300_000_000_000) * 1103515245 + 12345) % 1000
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        signal = dataframe["random_bucket"] < 10
        dataframe.loc[signal, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        signal = dataframe["random_bucket"] > 980
        dataframe.loc[signal, "exit_long"] = 1
        return dataframe
