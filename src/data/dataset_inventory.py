from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_metadata(metadata_dir: Path) -> list[dict[str, Any]]:
    if not metadata_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for path in sorted(metadata_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_metadata_path"] = str(path)
        records.append(record)
    return records


def assert_required_candles(
    records: list[dict[str, Any]],
    *,
    symbol: str,
    timeframes: list[str],
) -> None:
    available = {
        record.get("timeframe")
        for record in records
        if record.get("exchange") == "binance"
        and record.get("market_type") == "spot"
        and record.get("symbol") == symbol.upper()
        and record.get("validation_status") == "passed"
    }
    missing = sorted(set(timeframes) - available)
    if missing:
        raise ValueError(f"Missing validated candle datasets for {symbol.upper()}: {missing}")
