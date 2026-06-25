from __future__ import annotations

from freqtrade.strategy import IStrategy
from pandas import DataFrame


class RsiMeanReversionBaseline(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 80
    minimal_roi = {"0": 0.02, "180": 0.008, "480": 0}
    stoploss = -0.03
    trailing_stop = False
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        delta = dataframe["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))

        dataframe["rsi"] = 100 - (100 / (1 + rs))
        dataframe["mean_price"] = dataframe["close"].rolling(50, min_periods=50).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        signal = (
            (dataframe["rsi"] < 30)
            & (dataframe["close"] < dataframe["mean_price"])
            & (dataframe["rsi"] > dataframe["rsi"].shift(1))
        )
        dataframe.loc[signal, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        signal = (dataframe["rsi"] > 55) | (dataframe["close"] > dataframe["mean_price"])
        dataframe.loc[signal, "exit_long"] = 1
        return dataframe
