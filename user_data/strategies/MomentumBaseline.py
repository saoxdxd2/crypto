from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IStrategy


class MomentumBaseline(IStrategy):
    """
    Phase 2: Baseline Strategy - Momentum (ROC / MACD)
    Used to benchmark TimesFM. Trades based on positive MACD momentum.
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    minimal_roi = {
        "0": 0.05,
        "60": 0.02,
        "120": 0
    }
    
    stoploss = -0.05

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Enter when MACD crosses above signal line (Positive Momentum)
        dataframe.loc[
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1)) &
            (dataframe['macdhist'] > 0) &
            (dataframe['volume'] > 0),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit when MACD crosses below signal line (Losing Momentum)
        dataframe.loc[
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1)) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1
        return dataframe
