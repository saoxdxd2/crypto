from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IStrategy


class RsiBaseline(IStrategy):
    """
    Phase 2: Baseline Strategy for Edge Verification.
    Used to benchmark TimesFM. If TimesFM cannot beat this simple RSI mean reversion
    strategy, the Promotion Gate will hard-block the deployment.
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    # Minimal ROI
    minimal_roi = {
        "0": 0.05,
        "30": 0.02,
        "60": 0.01,
        "120": 0
    }
    
    stoploss = -0.05

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI Mean Reversion
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] < 30) &  # Oversold
            (dataframe['volume'] > 0), # Ensure volume exists
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] > 70) &  # Overbought
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1
        return dataframe
