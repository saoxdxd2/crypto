from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from crypto_research.hashing import sha256_file, sha256_text
from crypto_research.schemas import CANDLE_COLUMNS, validate_candles

BINANCE_KLINE_COLUMNS = [
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def read_binance_kline_zip(path: Path, *, symbol: str, timeframe: str) -> pl.DataFrame:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected exactly one CSV in {path}, found {csv_names}")
        with archive.open(csv_names[0]) as csv_file:
            raw = pl.read_csv(
                csv_file,
                has_header=False,
                new_columns=BINANCE_KLINE_COLUMNS,
                schema_overrides={
                    "open_time_ms": pl.Int64,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "close_time_ms": pl.Int64,
                    "quote_volume": pl.Float64,
                    "trade_count": pl.Int64,
                },
            )

    return raw.select(
        pl.lit("binance").alias("exchange"),
        pl.lit(symbol.upper()).alias("symbol"),
        pl.lit(timeframe).alias("timeframe"),
        pl.from_epoch("open_time_ms", time_unit="ms").alias("open_time"),
        pl.from_epoch("close_time_ms", time_unit="ms").alias("close_time"),
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        pl.lit(True).alias("is_closed"),
        pl.lit("binance_public_data").alias("source"),
    ).select(CANDLE_COLUMNS)


def import_candle_zips(
    *,
    zip_paths: list[Path],
    symbol: str,
    timeframe: str,
    output_dir: Path,
    metadata_dir: Path,
) -> tuple[Path, Path]:
    if not zip_paths:
        raise ValueError("No candle zip files found to import")

    frames = [
        read_binance_kline_zip(path, symbol=symbol, timeframe=timeframe)
        for path in sorted(zip_paths)
    ]
    normalized = validate_candles(pl.concat(frames).sort("open_time"))

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    start = normalized["open_time"].min()
    end = normalized["close_time"].max()
    start_text = _isoformat(start)
    end_text = _isoformat(end)
    parquet_path = output_dir / f"candles_{start_text[:10]}_{end_text[:10]}.parquet"
    normalized.write_parquet(parquet_path)

    file_hashes = {path.name: sha256_file(path) for path in sorted(zip_paths)}
    dataset_hash = sha256_text(
        json.dumps(
            {
                "parquet_hash": sha256_file(parquet_path),
                "source_files": file_hashes,
            },
            sort_keys=True,
        )
    )
    metadata = {
        "dataset_hash": dataset_hash,
        "exchange": "binance",
        "symbol": symbol.upper(),
        "market_type": "spot",
        "timeframe": timeframe,
        "start": start_text,
        "end": end_text,
        "source": "binance_public_data",
        "created_at": datetime.now(UTC).isoformat(),
        "validation_status": "passed",
        "source_file_hashes": file_hashes,
        "row_count": normalized.height,
    }
    metadata_path = metadata_dir / f"candles_binance_spot_{symbol.upper()}_{timeframe}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return parquet_path, metadata_path


def _isoformat(value: object) -> str:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC).isoformat()
    return str(value)
