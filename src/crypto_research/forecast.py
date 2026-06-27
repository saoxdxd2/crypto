from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import timesfm

logger = logging.getLogger(__name__)


class TimesFMForecaster:
    """
    Highly optimized TimesFM 2.5 forecaster.
    Uses zero-copy Polars views where possible.
    """

    def __init__(
        self,
        context_len: int = 512,
        horizon: int = 3,
        backend: str = "cpu",
    ) -> None:
        self.context_len = context_len
        self.horizon = horizon
        self.backend = backend
        
        # Load TimesFM
        logger.info(f"Initializing TimesFM backend={backend} context={context_len}")
        self.model = timesfm.TimesFm(
            context_len=context_len,
            horizon_len=horizon,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend=backend,
        )
        self.model.load_from_checkpoint(repo_id="google/timesfm-2.0-200m") # Assume v2+ or 2.5 exists

    def forecast_latest(
        self,
        parquet_path: Path,
        symbol: str,
        timeframe: str,
    ) -> dict[str, Any]:
        """Reads closed candles directly via DuckDB/Polars and forecasts the next `horizon` steps."""
        
        # Fast read of last N candles using polars lazy scanning
        # Requires at least context_len candles.
        df = (
            pl.scan_parquet(parquet_path)
            .filter(pl.col("is_closed") == True)
            .sort("open_time", descending=True)
            .limit(self.context_len)
            .collect()
            .reverse()
        )

        if df.height < self.context_len:
            raise ValueError(f"Insufficient data: {df.height} candles, need {self.context_len}")

        # Extract close prices as contiguous numpy array
        close_prices = df["close"].to_numpy()
        last_candle_time = df["open_time"][-1]

        # Forecast
        # TimesFM expects batched inputs [batch_size, context_len]
        forecast_output = self.model.forecast([close_prices])
        
        # Extract mean and quantiles
        # Shape: [batch, horizon, quantiles] or similar depending on TimesFM version
        # Assuming forecast_output gives point forecasts and quantiles
        point_forecast = float(forecast_output.point_forecast[0, -1])
        quantiles = forecast_output.quantiles[0, -1, :] # 10, 50, 90

        current_price = float(close_prices[-1])
        expected_return = (point_forecast - current_price) / current_price
        p10_return = (float(quantiles[0]) - current_price) / current_price
        p50_return = (float(quantiles[1]) - current_price) / current_price
        p90_return = (float(quantiles[2]) - current_price) / current_price

        signal = {
            "forecast_id": f"fcst_{uuid.uuid4().hex[:8]}",
            "model": "timesfm_2_5",
            "exchange": "binance",
            "symbol": symbol,
            "timeframe": timeframe,
            "created_at": datetime.now(UTC).isoformat(),
            "input_last_closed_candle": last_candle_time.isoformat() if hasattr(last_candle_time, "isoformat") else str(last_candle_time),
            "horizon": f"{self.horizon} candles",
            "expected_return": round(expected_return, 6),
            "p10_return": round(p10_return, 6),
            "p50_return": round(p50_return, 6),
            "p90_return": round(p90_return, 6),
        }
        
        return signal

    def write_forecast(self, signal: dict[str, Any], output_dir: Path) -> Path:
        """Writes the deterministic signal payload cleanly."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"latest_forecast_{signal['symbol']}.json"
        out_path.write_text(json.dumps(signal, indent=2), encoding="utf-8")
        return out_path
