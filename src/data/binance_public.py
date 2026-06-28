from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import requests

BINANCE_PUBLIC_DATA_URL = "https://data.binance.vision"


@dataclass(frozen=True)
class BinanceCandleFile:
    symbol: str
    timeframe: str
    day: date
    market_type: str = "spot"

    @property
    def filename(self) -> str:
        return f"{self.symbol.upper()}-{self.timeframe}-{self.day.isoformat()}.zip"

    @property
    def relative_path(self) -> str:
        return (
            f"data/{self.market_type}/daily/klines/"
            f"{self.symbol.upper()}/{self.timeframe}/{self.filename}"
        )

    @property
    def url(self) -> str:
        return f"{BINANCE_PUBLIC_DATA_URL}/{self.relative_path}"

    @property
    def checksum_url(self) -> str:
        return f"{self.url}.CHECKSUM"


def iter_days(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end must be on or after start")

    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def download(url: str, destination: Path, timeout_seconds: int = 60) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout_seconds)
    if response.status_code == 404:
        return False
    response.raise_for_status()
    destination.write_bytes(response.content)
    return True


def read_checksum(checksum_path: Path) -> str | None:
    if not checksum_path.exists():
        return None
    text = checksum_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return text.split()[0].lower()


def verify_sha256(path: Path, expected_hash: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != expected_hash.lower():
        raise ValueError(f"Checksum mismatch for {path}: expected {expected_hash}, got {digest}")


def download_daily_candles(
    *,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    output_dir: Path,
    verify_checksums: bool = True,
) -> list[Path]:
    downloaded: list[Path] = []

    for day in iter_days(start, end):
        remote_file = BinanceCandleFile(symbol=symbol, timeframe=timeframe, day=day)
        zip_path = output_dir / remote_file.filename
        checksum_path = output_dir / f"{remote_file.filename}.CHECKSUM"

        if not zip_path.exists():
            if not download(remote_file.url, zip_path):
                continue

        if verify_checksums:
            if not checksum_path.exists():
                download(remote_file.checksum_url, checksum_path)
            expected_hash = read_checksum(checksum_path)
            if expected_hash:
                verify_sha256(zip_path, expected_hash)

        downloaded.append(zip_path)

    return downloaded
