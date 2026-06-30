from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from crypto_research.baselines.catalog import baseline_catalog
from src.data.binance_public import download_daily_candles
from src.data.dataset_inventory import assert_required_candles, load_metadata
from src.core.duckdb_queries import query_candles
from src.data.ingest import import_candle_zips
from src.core.paths import DataPaths

app = typer.Typer(no_args_is_help=True)


def _paths(data_root: Path) -> DataPaths:
    return DataPaths(root=data_root)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise typer.BadParameter("Use YYYY-MM-DD format") from error


@app.command()
def init_dirs(data_root: Annotated[Path, typer.Option()] = Path("data")) -> None:
    paths = _paths(data_root)
    paths.ensure_phase1_dirs()
    typer.echo(f"Initialized data directories under {paths.root}")


@app.command()
def download_candles(
    symbol: Annotated[str, typer.Option()] = "BTCUSDT",
    timeframe: Annotated[str, typer.Option()] = "1m",
    start: Annotated[str, typer.Option()] = ...,
    end: Annotated[str, typer.Option()] = ...,
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    verify_checksums: Annotated[bool, typer.Option()] = True,
) -> None:
    paths = _paths(data_root)
    output_dir = paths.candle_zip_dir(symbol, timeframe)
    downloaded = download_daily_candles(
        symbol=symbol,
        timeframe=timeframe,
        start=_parse_date(start),
        end=_parse_date(end),
        output_dir=output_dir,
        verify_checksums=verify_checksums,
    )
    typer.echo(json.dumps({"downloaded": [str(path) for path in downloaded]}, indent=2))


@app.command()
def prepare_candles(
    symbol: Annotated[str, typer.Option()] = "BTCUSDT",
    timeframes: Annotated[list[str] | None, typer.Option()] = None,
    start: Annotated[str, typer.Option()] = ...,
    end: Annotated[str, typer.Option()] = ...,
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    verify_checksums: Annotated[bool, typer.Option()] = True,
) -> None:
    paths = _paths(data_root)
    paths.ensure_phase1_dirs()
    selected_timeframes = timeframes or ["1m", "5m", "15m"]

    prepared = []
    for timeframe in selected_timeframes:
        zip_paths = download_daily_candles(
            symbol=symbol,
            timeframe=timeframe,
            start=_parse_date(start),
            end=_parse_date(end),
            output_dir=paths.candle_zip_dir(symbol, timeframe),
            verify_checksums=verify_checksums,
        )
        parquet_path, metadata_path = import_candle_zips(
            zip_paths=zip_paths,
            symbol=symbol,
            timeframe=timeframe,
            output_dir=paths.candle_parquet_dir(symbol, timeframe),
            metadata_dir=paths.metadata,
        )
        prepared.append(
            {
                "timeframe": timeframe,
                "source_files": [str(path) for path in zip_paths],
                "parquet": str(parquet_path),
                "metadata": str(metadata_path),
            }
        )

    assert_required_candles(
        load_metadata(paths.metadata),
        symbol=symbol,
        timeframes=selected_timeframes,
    )
    typer.echo(json.dumps({"prepared": prepared}, indent=2))


@app.command()
def import_candles(
    symbol: Annotated[str, typer.Option()] = "BTCUSDT",
    timeframe: Annotated[str, typer.Option()] = "1m",
    data_root: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    paths = _paths(data_root)
    zip_paths = sorted(paths.candle_zip_dir(symbol, timeframe).glob("*.zip"))
    parquet_path, metadata_path = import_candle_zips(
        zip_paths=zip_paths,
        symbol=symbol,
        timeframe=timeframe,
        output_dir=paths.candle_parquet_dir(symbol, timeframe),
        metadata_dir=paths.metadata,
    )
    typer.echo(json.dumps({"parquet": str(parquet_path), "metadata": str(metadata_path)}, indent=2))


@app.command()
def inventory(
    data_root: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    paths = _paths(data_root)
    typer.echo(json.dumps(load_metadata(paths.metadata), indent=2, default=str))


@app.command(name="baseline-catalog")
def baseline_catalog_command() -> None:
    typer.echo(json.dumps(baseline_catalog(), indent=2))


@app.command(name="query-candles")
def query_candles_command(
    symbol: Annotated[str, typer.Option()] = "BTCUSDT",
    timeframe: Annotated[str, typer.Option()] = "1m",
    limit: Annotated[int, typer.Option(min=1)] = 10,
    data_root: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    paths = _paths(data_root)
    rows = query_candles(
        parquet_dir=paths.candle_parquet_dir(symbol, timeframe),
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )
    typer.echo(json.dumps(rows, indent=2, default=str))
