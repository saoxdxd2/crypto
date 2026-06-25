from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from crypto_research.schemas import validate_candles


def valid_candles() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "exchange": ["binance", "binance"],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "timeframe": ["1m", "1m"],
            "open_time": [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 1)],
            "close_time": [datetime(2024, 1, 1, 0, 0, 59), datetime(2024, 1, 1, 0, 1, 59)],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1.0, 2.0],
            "quote_volume": [100.0, 200.0],
            "trade_count": [10, 20],
            "is_closed": [True, True],
            "source": ["binance_public_data", "binance_public_data"],
        }
    )


def test_validate_candles_accepts_valid_closed_candles() -> None:
    assert validate_candles(valid_candles()).height == 2


def test_validate_candles_rejects_bad_ohlc() -> None:
    df = valid_candles().with_columns(pl.lit(200.0).alias("low"))

    with pytest.raises(ValueError, match="OHLC"):
        validate_candles(df)


def test_validate_candles_rejects_incomplete_candle() -> None:
    df = valid_candles().with_columns(pl.lit(False).alias("is_closed"))

    with pytest.raises(ValueError, match="closed"):
        validate_candles(df)
