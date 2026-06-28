from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path = Path("data")

    @property
    def raw_binance_spot(self) -> Path:
        return self.root / "raw" / "binance" / "spot"

    @property
    def normalized_candles(self) -> Path:
        return self.root / "normalized" / "candles"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata"

    def candle_zip_dir(self, symbol: str, timeframe: str) -> Path:
        return self.raw_binance_spot / symbol.upper() / "klines" / timeframe

    def candle_parquet_dir(self, symbol: str, timeframe: str) -> Path:
        return self.normalized_candles / "exchange=binance" / "market_type=spot" / (
            f"symbol={symbol.upper()}"
        ) / f"timeframe={timeframe}"

    def ensure_phase1_dirs(self) -> None:
        for path in [
            self.raw_binance_spot,
            self.normalized_candles,
            self.root / "raw" / "binance" / "spot" / "BTCUSDT" / "trades",
            self.root / "raw" / "binance" / "spot" / "BTCUSDT" / "depth",
            self.root / "raw" / "binance" / "spot" / "BTCUSDT" / "book_ticker",
            self.root / "features",
            self.root / "signals",
            self.root / "reports",
            self.root / "mlflow",
            self.metadata,
        ]:
            path.mkdir(parents=True, exist_ok=True)
