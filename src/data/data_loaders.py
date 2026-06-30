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
            raise FileNotFoundError(f"Dataset {self.parquet_path} not found. The harness should have pre-seeded this!")
            
        self.df = pl.read_parquet(self.parquet_path)
            
        assert len(self.df) > seq_len, "Not enough data in Parquet file."
        self.data_tensor = torch.tensor(
            self.df[["price", "volume", "side", "order_type"]].to_numpy(), 
            dtype=torch.float32
        )
        timestamps = self.df["timestamp_ms"].to_numpy()
        self.timestamps_tensor = torch.tensor(timestamps - timestamps[0], dtype=torch.float32)
        self.num_samples = len(self.df) - self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Zero-copy tensor slicing (virtually instantaneous)
        messages = self.data_tensor[idx:idx+self.seq_len]
        ts_seq = self.timestamps_tensor[idx:idx+self.seq_len]
        
        # Target: probability of price going up (1.0 if next_price > current_price else 0.0)
        current_price = self.data_tensor[idx+self.seq_len-1, 0]
        next_price = self.data_tensor[idx+self.seq_len, 0]
        
        # Adding tiny epsilon to prevent division by zero in weird edge cases
        raw_return = (next_price - current_price) / (current_price + 1e-8)
        target = torch.tensor(1.0 if raw_return > 0 else 0.0, dtype=torch.float32)
        
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
            raise FileNotFoundError(f"Dataset {self.parquet_path} not found. The harness should have pre-seeded this!")
            
        self.df = pl.read_parquet(self.parquet_path).filter(pl.col("is_closed") == True)
            
        self.data_tensor = torch.tensor(
            self.df[["open", "high", "low", "close", "volume"]].to_numpy(),
            dtype=torch.float32
        )
        self.num_samples = len(self.df) - self.seq_len - 1 # Need +1 for the target
        self.is_dummy = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.is_dummy:
            x = torch.rand(self.seq_len, 5)
            y = torch.rand(1).squeeze(-1) 
            return x, y
            
        # Zero-copy tensor slicing
        x = self.data_tensor[idx:idx+self.seq_len]
        
        # Target is the return of the close price on the next candle
        current_close = self.data_tensor[idx+self.seq_len-1, 3]
        next_close = self.data_tensor[idx+self.seq_len, 3]
        target_return = (next_close - current_close) / (current_close + 1e-8)
        
        y = torch.tensor(target_return, dtype=torch.float32)
        return x, y
