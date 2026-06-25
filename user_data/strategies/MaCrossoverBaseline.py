from __future__ import annotations

from freqtrade.strategy import IStrategy
from pandas import DataFrame


class MaCrossoverBaseline(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 60
    minimal_roi = {"0": 0.03, "120": 0.01, "360": 0}
    stoploss = -0.04
    trailing_stop = False
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ma_fast"] = dataframe["close"].rolling(20, min_periods=20).mean()
        dataframe["ma_slow"] = dataframe["close"].rolling(50, min_periods=50).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        crossed_up = (dataframe["ma_fast"] > dataframe["ma_slow"]) & (
            dataframe["ma_fast"].shift(1) <= dataframe["ma_slow"].shift(1)
        )
        dataframe.loc[crossed_up, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        crossed_down = (dataframe["ma_fast"] < dataframe["ma_slow"]) & (
            dataframe["ma_fast"].shift(1) >= dataframe["ma_slow"].shift(1)
        )
        dataframe.loc[crossed_down, "exit_long"] = 1
        return dataframe
