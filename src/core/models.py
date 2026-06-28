import json
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
from pydantic_settings import BaseSettings

class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"

class TradeRecord(BaseModel):
    id: str
    time: str
    symbol: str
    side: TradeSide
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0
    net_edge: float

class SignalRecord(BaseModel):
    timestamp: int
    symbol: str
    pattern: str
    confidence: float
    book_imbalance: float

class AppConfig(BaseSettings):
    symbol: str = "BTCUSDT"
    trade_size: float = 0.1
    stop_loss_usd: float = 15.0
    take_profit_usd: float = 15.0
    random_exit_prob: float = 0.02
    momentum_threshold: float = 10.0
    llm_server_url: str = "http://localhost:5001/generate"
    data_dir: Path = Path("data")
    
    class Config:
        env_prefix = "CRYPTO_"

config = AppConfig()
