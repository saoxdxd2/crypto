import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class L2Collector:
    """
    Phase 7: Binance L2 Order Book Collector
    Collects trades, depth, and bookTicker events live from Binance WebSockets.
    """

    def __init__(self, symbol: str = "btcusdt", batch_size: int = 5000) -> None:
        self.symbol = symbol.lower()
        self.batch_size = batch_size
        self.stream_url = (
            f"wss://stream.binance.com:9443/stream?streams="
            f"{self.symbol}@trade/{self.symbol}@depth@100ms/{self.symbol}@bookTicker"
        )
        self.buffer: list[dict[str, Any]] = []
        self.last_update_id = -1
        
        # Output paths
        self.base_dir = Path("data/raw/binance/spot") / self.symbol.upper()
        self.trades_dir = self.base_dir / "trades"
        self.depth_dir = self.base_dir / "depth"
        self.book_dir = self.base_dir / "book_ticker"
        
        for p in [self.trades_dir, self.depth_dir, self.book_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def _detect_sequence_gap(self, current_u: int, previous_U: int) -> None:
        """Detect sequence gaps for orderbook depth stream (u = final, U = first)"""
        if self.last_update_id != -1 and previous_U > self.last_update_id + 1:
            logger.warning(
                f"SEQUENCE GAP DETECTED! Missed updates between {self.last_update_id} and {previous_U}"
            )
        self.last_update_id = current_u

    def _flush_buffer(self) -> None:
        if not self.buffer:
            return

        # Separate by event type
        trades, depths, books = [], [], []
        for item in self.buffer:
            ev_type = item.get("event_type")
            if ev_type == "trade":
                trades.append(item)
            elif ev_type == "depthUpdate":
                depths.append(item)
            elif ev_type == "bookTicker":
                books.append(item)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")

        if trades:
            pl.DataFrame(trades).write_parquet(self.trades_dir / f"trades_{timestamp}.parquet")
        if depths:
            pl.DataFrame(depths).write_parquet(self.depth_dir / f"depth_{timestamp}.parquet")
        if books:
            pl.DataFrame(books).write_parquet(self.book_dir / f"book_{timestamp}.parquet")

        logger.info(f"Flushed batch: {len(trades)} trades, {len(depths)} depth, {len(books)} books.")
        self.buffer.clear()

    async def _process_message(self, message: str) -> None:
        receive_time = datetime.now(UTC).isoformat()
        
        try:
            payload = json.loads(message)
            if "data" not in payload:
                return

            data = payload["data"]
            event_type = data.get("e", "bookTicker") # bookTicker lacks 'e' field sometimes
            
            normalized_event = {
                "exchange": "binance",
                "symbol": self.symbol.upper(),
                "market_type": "spot",
                "event_type": event_type,
                "event_time": data.get("E", 0),
                "receive_time": receive_time,
                "raw_payload_hash": hash(message) % 1000000000, # Simple hash for demo
                "source": "binance_websocket"
            }

            if event_type == "trade":
                normalized_event.update({
                    "price": float(data["p"]),
                    "quantity": float(data["q"]),
                    "side": "sell" if data["m"] else "buy", # m = is the buyer the market maker
                    "trade_id": data["t"]
                })
            elif event_type == "depthUpdate":
                # U: first update ID, u: final update ID
                self._detect_sequence_gap(data["u"], data["U"])
                normalized_event.update({
                    "bids": json.dumps(data["b"]),
                    "asks": json.dumps(data["a"]),
                    "first_update_id": data["U"],
                    "final_update_id": data["u"]
                })
            elif event_type == "bookTicker":
                # BookTicker has u (updateId), b (best bid), B (bid qty), a (best ask), A (ask qty)
                normalized_event.update({
                    "best_bid": float(data["b"]),
                    "best_bid_qty": float(data["B"]),
                    "best_ask": float(data["a"]),
                    "best_ask_qty": float(data["A"]),
                    "update_id": data["u"]
                })

            self.buffer.append(normalized_event)

            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def start_collecting(self) -> None:
        logger.info(f"Starting Binance L2 Collector for {self.symbol.upper()}")
        
        while True:
            try:
                async with websockets.connect(self.stream_url) as websocket:
                    logger.info("Connected to Binance WebSocket.")
                    while True:
                        msg = await websocket.recv()
                        await self._process_message(str(msg))
            except ConnectionClosed:
                logger.warning("WebSocket Connection Closed. Reconnecting in 5 seconds...")
                self._flush_buffer()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected WebSocket error: {e}. Reconnecting in 5 seconds...")
                self._flush_buffer()
                await asyncio.sleep(5)
