from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IStrategy


class MaCrossover(IStrategy):
    """
    Phase 2: Baseline Strategy - Moving Average Crossover
    Used to benchmark TimesFM. Simple Fast MA crossing over Slow MA.
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    minimal_roi = {
        "0": 0.05,
        "30": 0.02,
        "60": 0.01,
        "120": 0
    }
    
    stoploss = -0.05

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['fast_ma'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['slow_ma'] = ta.SMA(dataframe, timeperiod=50)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Enter when Fast MA crosses above Slow MA
        dataframe.loc[
            (dataframe['fast_ma'] > dataframe['slow_ma']) &
            (dataframe['fast_ma'].shift(1) <= dataframe['slow_ma'].shift(1)) &
            (dataframe['volume'] > 0),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit when Fast MA crosses below Slow MA
        dataframe.loc[
            (dataframe['fast_ma'] < dataframe['slow_ma']) &
            (dataframe['fast_ma'].shift(1) >= dataframe['slow_ma'].shift(1)) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1
        return dataframe
