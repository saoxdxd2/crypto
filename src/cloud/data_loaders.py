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
        
        # Load directly into memory if it fits, else this should use LazyFrame batching
        if not self.parquet_path.exists():
            # Create a dummy dataset for demo purposes if no file exists
            logger.warning(f"File {self.parquet_path} not found. Generating dummy data for training demo.")
            self.num_samples = 10000
            self.is_dummy = True
        else:
            self.df = pl.read_parquet(self.parquet_path)
            # Ensure we have enough data
            assert len(self.df) > seq_len, "Not enough data in Parquet file."
            # Extract features as numpy
            # Expected columns: price, volume, side, order_type, timestamp_ms
            self.data = self.df[["price", "volume", "side", "order_type"]].to_numpy()
            self.timestamps = self.df["timestamp_ms"].to_numpy()
            self.num_samples = len(self.df) - self.seq_len
            self.is_dummy = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.is_dummy:
            messages = torch.rand(self.seq_len, 4)
            timestamps = torch.cumsum(torch.randint(1, 50, (self.seq_len,)), dim=0)
            return messages, timestamps
            
        # Get sequence
        messages = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.float32)
        
        # Get timestamps and calculate deltas or absolute relative to sequence start
        ts_seq = torch.tensor(self.timestamps[idx:idx+self.seq_len], dtype=torch.float32)
        # Normalize timestamps relative to the first event in the sequence
        ts_seq = ts_seq - ts_seq[0] 
        
        return messages, ts_seq


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
