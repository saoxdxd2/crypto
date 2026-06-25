from __future__ import annotations

from pathlib import Path

import duckdb


def query_candles(
    *,
    parquet_dir: Path,
    symbol: str,
    timeframe: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    pattern = str(parquet_dir / "*.parquet").replace("\\", "/")
    query = """
        select
            exchange,
            symbol,
            timeframe,
            open_time,
            close_time,
            open,
            high,
            low,
            close,
            volume,
            quote_volume,
            trade_count,
            is_closed,
            source
        from read_parquet(?)
        where symbol = ? and timeframe = ?
        order by open_time
        limit ?
    """
    with duckdb.connect() as connection:
        rows = connection.execute(query, [pattern, symbol.upper(), timeframe, limit]).fetchall()
        columns = [description[0] for description in connection.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]
