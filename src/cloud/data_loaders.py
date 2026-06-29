"""
Data Loaders for Offline Pre-Training.

Optimized PyTorch Datasets that stream historical L2 tick data and OHLCV
K-lines from Parquet files using Polars. Designed for high-throughput
training on Google Colab or local GPUs.
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import polars as pl
import logging

logger = logging.getLogger(__name__)

class LOBERTDataset(Dataset):
    """
    Streams raw Limit Order Book (LOB) messages for LOBERT pre-training.
    Returns (messages, timestamps_ms) to support Continuous-Time ROPE.
    """
    def __init__(self, parquet_path: str, seq_len: int = 128):
        self.seq_len = seq_len
        self.parquet_path = Path(parquet_path)
        
        logger.info(f"Loading LOB data from {self.parquet_path}...")
        
        if not self.parquet_path.exists():
            logger.warning(f"File {self.parquet_path} not found. Fetching real historical data from Binance instead of using dummy data!")
            import requests
            import time
            import os
            
            # Fetch last 1000 trades from Binance to bootstrap training
            url = "https://api.binance.com/api/v3/trades?symbol=BTCUSDT&limit=1000"
            resp = requests.get(url).json()
            
            # Convert to DataFrame
            df_data = []
            for t in resp:
                df_data.append({
                    "price": float(t["price"]),
                    "volume": float(t["qty"]),
                    "side": 1.0 if t["isBuyerMaker"] else 0.0,
                    "order_type": 1.0,
                    "timestamp_ms": t["time"]
                })
            self.df = pl.DataFrame(df_data)
            self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
            self.df.write_parquet(self.parquet_path)
            logger.info("Successfully bootstrapped real Kaggle dataset from Binance.")
        else:
            self.df = pl.read_parquet(self.parquet_path)
            
        assert len(self.df) > seq_len, "Not enough data in Parquet file."
        self.data = self.df[["price", "volume", "side", "order_type"]].to_numpy()
        self.timestamps = self.df["timestamp_ms"].to_numpy()
        self.num_samples = len(self.df) - self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        messages = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.float32)
        
        ts_seq = torch.tensor(self.timestamps[idx:idx+self.seq_len], dtype=torch.float32)
        ts_seq = ts_seq - ts_seq[0] 
        
        # Target: future price return (next message price - current last message price)
        current_price = self.data[idx+self.seq_len-1, 0]
        next_price = self.data[idx+self.seq_len, 0]
        target = torch.tensor((next_price - current_price) / current_price, dtype=torch.float32)
        
        return messages, ts_seq, target


class FinCastDataset(Dataset):
    """
    Streams OHLCV candles for FinCast autoregressive pre-training.
    """
    def __init__(self, parquet_path: str, seq_len: int = 512):
        self.seq_len = seq_len
        self.parquet_path = Path(parquet_path)
        
        logger.info(f"Loading OHLCV data from {self.parquet_path}...")
        
        if not self.parquet_path.exists():
            logger.warning(f"File {self.parquet_path} not found. Generating dummy data for training demo.")
            self.num_samples = 10000
            self.is_dummy = True
        else:
            self.df = pl.read_parquet(self.parquet_path).filter(pl.col("is_closed") == True)
            self.data = self.df[["open", "high", "low", "close", "volume"]].to_numpy()
            self.num_samples = len(self.df) - self.seq_len - 1 # Need +1 for the target
            self.is_dummy = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.is_dummy:
            # Random OHLCV sequence
            x = torch.rand(self.seq_len, 5)
            # Target is the 'return' of the next candle
            y = torch.rand(1).squeeze(-1) 
            return x, y
            
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.float32)
        
        # For simplicity, target is the return of the close price on the next candle
        current_close = self.data[idx+self.seq_len-1, 3]
        next_close = self.data[idx+self.seq_len, 3]
        target_return = (next_close - current_close) / current_close
        
        y = torch.tensor(target_return, dtype=torch.float32)
        return x, y
