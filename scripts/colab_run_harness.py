"""
Autonomous CPU Online Adaptation Harness.

This agentic script runs LOBERT and FinCast in a continuous, localized, CPU-only
online adaptation loop. It fetches historical data via the Data Agent, streams it,
and runs `finetune_step()` sequentially. No offline GPU/Cloud infrastructure required.
"""

import os
import sys
import json
import time
import threading
from pathlib import Path

# Try importing google.colab to use secrets, otherwise fallback to env
try:
    from google.colab import userdata, files, runtime
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def get_api_key():
    return "AQ.Ab8RN6Ifh1nrh3KyuJkTuKUC_FW6rvMipV1Bg6nieB_lDX1h-w"

class BinanceArchiveFetcher:
    """Downloads monthly archive ZIPs and streams to Parquet instantly."""
    
    BASE_URL = "https://data.binance.vision/data/spot/monthly"
    
    @staticmethod
    def fetch_trades(symbol: str, year: str, month: str, output_parquet: Path):
        month = str(month).zfill(2)
        filename = f"{symbol}-trades-{year}-{month}.zip"
        url = f"{BinanceArchiveFetcher.BASE_URL}/trades/{symbol}/{filename}"
        
        print(f"\n[Data Agent] 📥 Fetching Trades: {url}")
        extract_dir = Path("temp_data")
        extract_dir.mkdir(exist_ok=True)
        
        try:
            dl_cmd = f"aria2c -x 16 -s 16 -j 16 --continue=true -d {extract_dir} -o {filename} {url}"
            if os.system(dl_cmd) != 0:
                print(f"[Data Agent] ❌ aria2c download failed!")
                return False
                
            print(f"[Data Agent] ⚡ Extracting...")
            unzip_cmd = f"unzip -q -o {extract_dir}/{filename} -d {extract_dir}"
            if os.system(unzip_cmd) != 0:
                print(f"[Data Agent] ❌ unzip failed!")
                return False
                
        except Exception as e:
            print(f"[Data Agent] ❌ Download/Extraction failed: {e}")
            return False
            
        csv_file = extract_dir / filename.replace(".zip", ".csv")
            
        print(f"[Data Agent] ⚡ Streaming CSV to Parquet via Polars...")
        try:
            import polars as pl
            lf = pl.scan_csv(csv_file, has_header=False, new_columns=["id", "price", "volume", "quoteQty", "timestamp_ms", "isBuyerMaker", "isBestMatch"])
            lf = lf.with_columns([
                pl.when(pl.col("isBuyerMaker")).then(1.0).otherwise(0.0).alias("side"),
                pl.lit(1.0).alias("order_type")
            ])
            lf = lf.select(["price", "volume", "side", "order_type", "timestamp_ms"])
            lf.sink_parquet(output_parquet)
        except Exception as e:
            print(f"[Data Agent] ❌ Parsing failed: {e}")
            raise e
        
        if csv_file.exists():
            csv_file.unlink()
        zip_file_path = extract_dir / filename
        if zip_file_path.exists():
            zip_file_path.unlink()
        print(f"[Data Agent] ✅ Successfully processed {year}-{month} trades to {output_parquet}")
        return True

    @staticmethod
    def fetch_klines(symbol: str, year: str, month: str, output_parquet: Path):
        month = str(month).zfill(2)
        filename = f"{symbol}-1m-{year}-{month}.zip"
        url = f"{BinanceArchiveFetcher.BASE_URL}/klines/{symbol}/1m/{filename}"
        
        print(f"\n[Data Agent] 📥 Fetching Klines: {url}")
        extract_dir = Path("temp_data")
        extract_dir.mkdir(exist_ok=True)
        
        try:
            dl_cmd = f"aria2c -x 16 -s 16 -j 16 --continue=true -d {extract_dir} -o {filename} {url}"
            if os.system(dl_cmd) != 0:
                return False
                
            unzip_cmd = f"unzip -q -o {extract_dir}/{filename} -d {extract_dir}"
            if os.system(unzip_cmd) != 0:
                return False
                
        except Exception as e:
            return False
            
        csv_file = extract_dir / filename.replace(".zip", ".csv")
            
        try:
            import polars as pl
            lf = pl.scan_csv(csv_file, has_header=False, new_columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time", 
                "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            lf = lf.with_columns([
                pl.lit(True).alias("is_closed")
            ])
            lf = lf.select(["open_time", "open", "high", "low", "close", "volume", "is_closed"])
            lf.sink_parquet(output_parquet)
        except Exception as e:
            raise e
        
        if csv_file.exists():
            csv_file.unlink()
        zip_file_path = extract_dir / filename
        if zip_file_path.exists():
            zip_file_path.unlink()
        return True


def run_online_adaptation_loop():
    print("\n🚀 Initializing Online CPU Adaptation Loop...")
    import torch
    import numpy as np
    from src.core.lob_encoder import LOBERTModel
    from src.mission_control.forecast import FinCastForecaster
    from src.data.data_loaders import LOBERTDataset, FinCastDataset
    from torch.utils.data import DataLoader

    # Initialize CPU models
    lobert = LOBERTModel()
    fincast = FinCastForecaster()

    lob_path = Path("data/lob_history.parquet")
    ohlcv_path = Path("data/ohlcv_history.parquet")

    # Verify data exists
    if not lob_path.exists() or not ohlcv_path.exists():
        print("❌ Data not found! Fetching data first...")
        return False

    print("\n📦 Loading Datasets...")
    lob_dataset = LOBERTDataset(str(lob_path), seq_len=128)
    fincast_dataset = FinCastDataset(str(ohlcv_path), seq_len=512)

    lob_loader = DataLoader(lob_dataset, batch_size=32, shuffle=True)
    fincast_loader = DataLoader(fincast_dataset, batch_size=32, shuffle=True)

    print("\n🔥 Starting Continuous Fine-Tuning Steps...")
    
    # Run 50 steps for demonstration
    lobert_loss = 0
    for i, (messages, timestamps, targets) in enumerate(lob_loader):
        if i >= 50: break
        loss = lobert.finetune_step(messages, timestamps, targets)
        lobert_loss = loss
        if i % 10 == 0:
            print(f"[LOBERT] Step {i:03d} | Loss: {loss:.4f}")

    fincast_loss = 0
    for i, (x, y) in enumerate(fincast_loader):
        if i >= 50: break
        # FinCast finetune_step expects lists of numpy arrays
        ohlcv_windows = [win.numpy() for win in x]
        target_returns = [t.item() for t in y]
        loss = fincast.finetune_step(ohlcv_windows, target_returns)
        fincast_loss = loss
        if i % 10 == 0:
            print(f"[FinCast] Step {i:03d} | Loss: {loss:.4f}")

    print(f"\n🎉 Adaptation complete! LOBERT Loss: {lobert_loss:.4f} | FinCast Loss: {fincast_loss:.4f}")
    return True

def main():
    print("="*60)
    print("🤖 High-Efficiency CPU Online Adaptation Harness")
    print("="*60)

    try:
        repo_root = Path(__file__).resolve().parent.parent
        os.chdir(repo_root)
    except NameError:
        pass
    
    import shutil
    if IN_COLAB and not shutil.which("aria2c"):
        print("🚀 Installing aria2 and unzip for ultra-fast parallel fetching...")
        os.system("apt-get update && apt-get install -y aria2 unzip")

    Path("data").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Pre-seed datasets for loop 1
    def verify_parquet(path: Path) -> bool:
        if not path.exists(): return False
        try:
            with open(path, "rb") as f:
                f.seek(-4, 2)
                return f.read() == b"PAR1"
        except Exception:
            return False

    lob_path = Path("data/lob_history.parquet")
    ohlcv_path = Path("data/ohlcv_history.parquet")
    
    if not verify_parquet(lob_path):
        if lob_path.exists(): os.remove(lob_path)
        BinanceArchiveFetcher.fetch_trades("BTCUSDT", "2023", "01", lob_path)
    if not verify_parquet(ohlcv_path):
        if ohlcv_path.exists(): os.remove(ohlcv_path)
        BinanceArchiveFetcher.fetch_klines("BTCUSDT", "2023", "01", ohlcv_path)
        
    print("\n" + "="*60)
    print(f"🔄 LOOP 1/1 (CPU Continuous Adaptation)")
    print("="*60)
    
    success = run_online_adaptation_loop()
    
    if success:
        print("\n🏁 Online Adaptation Harness Complete.")

if __name__ == "__main__":
    main()
