from __future__ import annotations

from freqtrade.strategy import IStrategy
from pandas import DataFrame


class MomentumBaseline(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 80
    minimal_roi = {"0": 0.025, "180": 0.01, "480": 0}
    stoploss = -0.035
    trailing_stop = False
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["return_12"] = dataframe["close"].pct_change(12)
        dataframe["trend_ma"] = dataframe["close"].rolling(50, min_periods=50).mean()
        dataframe["volume_ma"] = dataframe["volume"].rolling(20, min_periods=20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        signal = (
            (dataframe["return_12"] > 0.006)
            & (dataframe["close"] > dataframe["trend_ma"])
            & (dataframe["volume"] > dataframe["volume_ma"])
        )
        dataframe.loc[signal, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        signal = (dataframe["return_12"] < 0) | (dataframe["close"] < dataframe["trend_ma"])
        dataframe.loc[signal, "exit_long"] = 1
        return dataframe
