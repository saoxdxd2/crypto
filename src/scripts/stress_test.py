import json
import time
import random
import os
from pathlib import Path

# Paths
SIG_DIR = Path("data/signals")
TRADES_DIR = Path("data/trades")
SIG_DIR.mkdir(parents=True, exist_ok=True)
TRADES_DIR.mkdir(parents=True, exist_ok=True)

SIG_FILE = SIG_DIR / "LOBERT_1m.json"
THINKING_FILE = SIG_DIR / "thinking_log.json"
TRADE_FILE = TRADES_DIR / "trade_history.json"

print("🔥 Initiating High-Frequency Stress Test...")
print("Flooding signals and trade events every 0.1s. Press Ctrl+C to stop.")

def generate_signal():
    return {
        "timestamp": int(time.time()),
        "symbol": "BTCUSDT",
        "pattern": random.choice(["BULL_FLAG", "BEAR_SWEEP", "CHOP"]),
        "confidence": random.uniform(0.1, 0.99),
        "book_imbalance": random.uniform(-1.0, 1.0)
    }

def generate_trade():
    from datetime import datetime, UTC
    pnl = random.uniform(-15.0, 30.0)
    return {
        "id": f"TEST_{random.randint(1000, 9999)}",
        "time": datetime.now(UTC).strftime("%H:%M:%S"),
        "symbol": "BTCUSDT",
        "side": random.choice(["LONG", "SHORT"]),
        "size": "0.1",
        "entry_price": 65000.0,
        "exit_price": 65000.0 + (pnl * 10),
        "pnl": pnl,
        "net_edge": round(random.uniform(0.001, 0.005), 4),
        "status": "CLOSED"
    }

def generate_thinking():
    from datetime import datetime, UTC
    return {
        "time": datetime.now(UTC).strftime("%H:%M:%S"),
        "verdict": random.choice(["ALLOW", "BLOCK", "REDUCE", "ALLOW"]),
        "reason": f"Stress test injected logic at edge={random.uniform(0.001, 0.005):.4f}",
        "progress": random.uniform(0, 1.0)
    }

try:
    counter = 0
    while True:
        # 1. Update Signals
        SIG_FILE.write_text(json.dumps([generate_signal()]))
        
        # 2. Update Trades (Appends to existing)
        trades = []
        if TRADE_FILE.exists():
            try:
                trades = json.loads(TRADE_FILE.read_text())
            except:
                pass
        trades.append(generate_trade())
        trades = trades[-50:] # Keep last 50
        TRADE_FILE.write_text(json.dumps(trades))
        
        # 3. Update Thinking
        thinking = []
        if THINKING_FILE.exists():
            try:
                thinking = json.loads(THINKING_FILE.read_text())
            except:
                pass
        thinking.append(generate_thinking())
        thinking = thinking[-20:]
        THINKING_FILE.write_text(json.dumps(thinking))
        
        counter += 1
        if counter % 10 == 0:
            print(f"[{counter}] Injected 10 highly-concurrent state mutations...")
            
        time.sleep(0.1) # 10 updates per second!

except KeyboardInterrupt:
    print("\n🛑 Stress Test Halted.")
