import numpy as np
from pandas import DataFrame
from freqtrade.strategy import IStrategy


class RandomBaseline(IStrategy):
    """
    Phase 2: Baseline Strategy - Random Entry
    Used as the absolute floor benchmark for TimesFM. 
    If TimesFM cannot beat a coin flip after fees, it is rejected.
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    minimal_roi = {
        "0": 0.05,
        "30": 0.02,
        "60": 0.01
    }
    
    stoploss = -0.05

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Generate a deterministic random series based on volume to keep backtests reproducible
        np.random.seed(42)
        dataframe['random_signal'] = np.random.uniform(0, 1, size=len(dataframe))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Enter long 5% of the time randomly
        dataframe.loc[
            (dataframe['random_signal'] > 0.95) &
            (dataframe['volume'] > 0),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long randomly or let ROI/Stoploss handle it.
        # We'll use a random 5% chance to exit to simulate holding periods.
        np.random.seed(43)
        dataframe['random_exit'] = np.random.uniform(0, 1, size=len(dataframe))
        
        dataframe.loc[
            (dataframe['random_exit'] > 0.95) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1
        return dataframe
