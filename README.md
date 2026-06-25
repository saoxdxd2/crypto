# Crypto Research

Phase 1 builds the free Binance data foundation. It does not trade, train models,
run LLM decision loops, or create a custom execution engine.

## Scope

- Binance public spot candle downloader.
- Checksum verification when Binance publishes a `.CHECKSUM` file.
- Parquet normalization for closed candles only.
- Pandera schema plus custom validation checks.
- Dataset metadata and hash generation.
- DuckDB query helpers for normalized Parquet.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e .[dev]
crypto-data init-dirs
crypto-data prepare-candles --symbol BTCUSDT --timeframes 1m --timeframes 5m --timeframes 15m --start 2024-01-01 --end 2024-01-02
crypto-data inventory
crypto-data query-candles --symbol BTCUSDT --timeframe 1m --limit 5
crypto-data baseline-catalog
```

All timestamps are UTC. The first supported target is `BTCUSDT` spot on Binance.

## Phase 2 Baselines

Freqtrade remains the only backtesting and dry-run execution shell in v1. Baseline
strategies live under `user_data/strategies`:

- `BuyAndHoldBaseline`
- `MaCrossoverBaseline`
- `MomentumBaseline`
- `RsiMeanReversionBaseline`
- `RandomEntryBaseline`

After installing Freqtrade, run:

```powershell
.\scripts\freqtrade_baselines.ps1 -Config config/freqtrade.dryrun.example.json -Timerange 20240101-20240102 -Timeframe 5m
```

Every baseline run must be paired with `freqtrade lookahead-analysis`. If any
strategy reports biased indicators or signals, reject it.
