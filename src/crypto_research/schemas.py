from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

import polars as pl
import pandera.polars as pa

CANDLE_COLUMNS: Final[list[str]] = [
    "exchange",
    "symbol",
    "timeframe",
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "is_closed",
    "source",
]


class CandleSchema(pa.DataFrameModel):
    exchange: str = pa.Field()
    symbol: str = pa.Field()
    timeframe: str = pa.Field()
    open_time: datetime = pa.Field()
    close_time: datetime = pa.Field()
    open: float = pa.Field(gt=0)
    high: float = pa.Field(gt=0)
    low: float = pa.Field(gt=0)
    close: float = pa.Field(gt=0)
    volume: float = pa.Field(ge=0)
    quote_volume: float = pa.Field(ge=0)
    trade_count: int = pa.Field(ge=0)
    is_closed: bool = pa.Field()
    source: str = pa.Field()

    @pa.dataframe_check
    @classmethod
    def check_open_lt_close_time(cls, df: Any) -> pl.LazyFrame:
        return df.lazyframe.select(pl.col("open_time") < pl.col("close_time"))

    @pa.dataframe_check
    @classmethod
    def check_ohlc_consistency(cls, df: Any) -> pl.LazyFrame:
        return df.lazyframe.select(
            (pl.col("low") <= pl.col("open")) &
            (pl.col("low") <= pl.col("high")) &
            (pl.col("low") <= pl.col("close")) &
            (pl.col("high") >= pl.col("open")) &
            (pl.col("high") >= pl.col("close"))
        )

    @pa.dataframe_check
    @classmethod
    def check_is_closed(cls, df: Any) -> pl.LazyFrame:
        return df.lazyframe.select(pl.col("is_closed") == True)

    @pa.dataframe_check
    @classmethod
    def check_future_time(cls, df: Any) -> pl.LazyFrame:
        now = datetime.now(UTC).replace(tzinfo=None)
        return df.lazyframe.select(pl.col("open_time") <= now)


def validate_candles(df: pl.DataFrame, *, allowed_gap_multiplier: int = 2) -> pl.DataFrame:
    if df.is_empty():
        raise ValueError("Candle dataset is empty")

    duplicate_count = (
        df.group_by(["exchange", "symbol", "timeframe", "open_time"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise ValueError("Duplicate candle open_time found per symbol/timeframe")
    
    validated_df = CandleSchema.validate(df)
    validate_candle_gaps(validated_df, allowed_gap_multiplier=allowed_gap_multiplier)
    return validated_df


def validate_candle_gaps(df: pl.DataFrame, *, allowed_gap_multiplier: int = 2) -> None:
    for _, partition in df.partition_by(
        ["exchange", "symbol", "timeframe"], as_dict=True, maintain_order=True
    ).items():
        ordered = partition.sort("open_time")
        if ordered.height < 2:
            continue
        diffs = ordered.select(pl.col("open_time").diff().alias("gap")).drop_nulls()
        expected_gap = diffs["gap"].min()
        if expected_gap is None:
            continue
        max_allowed_gap = expected_gap * allowed_gap_multiplier
        if diffs.filter(pl.col("gap") > max_allowed_gap).height:
            raise ValueError("Missing candles exceed allowed gap")
