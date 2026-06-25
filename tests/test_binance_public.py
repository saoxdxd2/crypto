from __future__ import annotations

from datetime import date

from crypto_research.binance_public import BinanceCandleFile, iter_days


def test_binance_daily_candle_url() -> None:
    candle_file = BinanceCandleFile(symbol="btcusdt", timeframe="1m", day=date(2024, 1, 2))

    assert candle_file.filename == "BTCUSDT-1m-2024-01-02.zip"
    assert (
        candle_file.url
        == "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01-02.zip"
    )
    assert candle_file.checksum_url.endswith(".zip.CHECKSUM")


def test_iter_days_inclusive() -> None:
    assert iter_days(date(2024, 1, 1), date(2024, 1, 3)) == [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
