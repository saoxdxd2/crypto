from __future__ import annotations

from freqtrade.strategy import IStrategy
from pandas import DataFrame


class BuyAndHoldBaseline(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 1
    minimal_roi = {"0": 100}
    stoploss = -0.99
    trailing_stop = False
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        if not dataframe.empty:
            dataframe.loc[dataframe.index[0], "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
