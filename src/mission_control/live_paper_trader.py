import time
import json
import logging
import asyncio
import websockets
import random
from datetime import datetime, UTC
from pathlib import Path
from filelock import FileLock
from src.core.sys_events import push_sys_event
from src.core.models import TradeRecord, SignalRecord, TradeSide, TradeStatus, config

from src.core.logger import logger

SIG_DIR = config.data_dir / "signals"
TRADES_DIR = config.data_dir / "trades"
SIG_DIR.mkdir(parents=True, exist_ok=True)
TRADES_DIR.mkdir(parents=True, exist_ok=True)

SIG_FILE = SIG_DIR / f"LOBERT_1m.json"
TRADE_FILE = TRADES_DIR / "trade_history.json"
REGIME_FILE = config.data_dir / "metadata" / "regime.json"

class LivePaperTrader:
    def __init__(self, symbol=config.symbol):
        self.symbol = symbol
        self.price_history = []
        self.active_trade: TradeRecord | None = None
        self.trade_counter = random.randint(1000, 9999)
        self.last_update = 0

    def _save_signal(self, pattern: str, confidence: float, edge: float):
        sig = SignalRecord(
            timestamp=int(time.time()),
            symbol=self.symbol,
            pattern=pattern,
            confidence=confidence,
            book_imbalance=edge
        )
        lock = FileLock(f"{SIG_FILE}.lock")
        with lock:
            signals = []
            if SIG_FILE.exists():
                try:
                    signals = json.loads(SIG_FILE.read_text())
                except:
                    pass
            signals.append(sig.model_dump())
            signals = signals[-100:] # Prevent OOM infinite disk growth
            SIG_FILE.write_text(json.dumps(signals))

    def _save_trade(self, trade: TradeRecord):
        lock = FileLock(f"{TRADE_FILE}.lock")
        with lock:
            trades = []
            if TRADE_FILE.exists():
                try:
                    trades = json.loads(TRADE_FILE.read_text())
                except:
                    pass
            # Update existing trade or append new
            updated = False
            for i, t in enumerate(trades):
                if t["id"] == trade.id:
                    trades[i] = trade.model_dump()
                    updated = True
                    break
            if not updated:
                trades.append(trade.model_dump())
            
            trades = trades[-50:] # Keep last 50
            TRADE_FILE.write_text(json.dumps(trades))

    async def _process_price(self, price: float):
        current_time = time.time()
        if current_time - self.last_update < 1.0:
            return
        
        self.last_update = current_time
        self.price_history.append(price)
        if len(self.price_history) > 10:
            self.price_history.pop(0)
        
        if len(self.price_history) >= 3:
            momentum = price - self.price_history[0]
            
            if not self.active_trade:
                # Decoupled LLM Safety Check
                regime_safe = True
                if REGIME_FILE.exists():
                    try:
                        lock = FileLock(f"{REGIME_FILE}.lock")
                        with lock:
                            regime_data = json.loads(REGIME_FILE.read_text())
                            regime_safe = regime_data.get("is_safe", True)
                    except:
                        pass
                
                if momentum > config.momentum_threshold:  # Bullish
                    if not regime_safe:
                        logger.info("BLOCKED LONG: LLM Regime Verifier marked market as unsafe.")
                        push_sys_event("BLOCK", "LLM Verifier blocked LONG entry due to regime safety.", progress=0.5)
                        self.last_update = time.time() + 10 # Backoff
                        return
                        
                    self._save_signal("BULL_SWEEP", 0.85, 0.05)
                    self.active_trade = TradeRecord(
                        id=f"PAPER_{self.trade_counter}",
                        time=datetime.now(UTC).strftime("%H:%M:%S"),
                        symbol=self.symbol,
                        side=TradeSide.LONG,
                        size=config.trade_size,
                        entry_price=price,
                        net_edge=0.05
                    )
                    self.trade_counter += 1
                    self._save_trade(self.active_trade)
                    logger.info(f"OPEN LONG at ${price:.2f}")
                    push_sys_event("ALLOW", f"Momentum breakout detected. OPEN LONG at ${price:.2f}", progress=0.5)
                    
                elif momentum < -config.momentum_threshold:  # Bearish
                    if not regime_safe:
                        logger.info("BLOCKED SHORT: LLM Regime Verifier marked market as unsafe.")
                        push_sys_event("BLOCK", "LLM Verifier blocked SHORT entry due to regime safety.", progress=0.5)
                        self.last_update = time.time() + 10 # Backoff
                        return
                        
                    self._save_signal("BEAR_SWEEP", 0.85, -0.05)
                    self.active_trade = TradeRecord(
                        id=f"PAPER_{self.trade_counter}",
                        time=datetime.now(UTC).strftime("%H:%M:%S"),
                        symbol=self.symbol,
                        side=TradeSide.SHORT,
                        size=config.trade_size,
                        entry_price=price,
                        net_edge=-0.05
                    )
                    self.trade_counter += 1
                    self._save_trade(self.active_trade)
                    logger.info(f"OPEN SHORT at ${price:.2f}")
                    push_sys_event("ALLOW", f"Momentum breakdown detected. OPEN SHORT at ${price:.2f}", progress=0.5)
            else:
                price_delta = price - self.active_trade.entry_price if self.active_trade.side == TradeSide.LONG else self.active_trade.entry_price - price
                actual_pnl = price_delta * self.active_trade.size
                if actual_pnl > config.take_profit_usd or actual_pnl < -config.stop_loss_usd or random.random() < config.random_exit_prob:
                    self.active_trade.exit_price = price
                    self.active_trade.pnl = round(actual_pnl, 2)
                    self.active_trade.status = TradeStatus.CLOSED
                    self._save_trade(self.active_trade)
                    logger.info(f"CLOSED {self.active_trade.side} at ${price:.2f} | PnL: ${actual_pnl:.2f}")
                    push_sys_event("REDUCE" if actual_pnl < 0 else "ALLOW", f"Trade closed. PnL: ${actual_pnl:.2f}", progress=1.0)
                    self.active_trade = None

    async def run_async(self):
        logger.info(f"Starting Async Live Paper Trading on {self.symbol}...")
        push_sys_event("SYSTEM", f"Paper Trader connected to Binance WebSocket for {self.symbol}", progress=0.1)
        
        ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@bookTicker"
        
        while True:
            try:
                async with websockets.connect(ws_url) as ws:
                    logger.info(f"Connected to {ws_url}")
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        best_ask = float(data.get('a', 0.0))
                        if best_ask > 0:
                            await self._process_price(best_ask)
            except Exception as e:
                logger.error(f"WebSocket Error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

def run_paper_trader():
    trader = LivePaperTrader()
    asyncio.run(trader.run_async())

if __name__ == "__main__":
    run_paper_trader()
