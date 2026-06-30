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
                
            print(f"[Data Agent] ⚡ Extracting massively via native C unzip...")
            import subprocess
            subprocess.run(["unzip", "-q", "-o", str(extract_dir / filename), "-d", str(extract_dir)], check=True)
            
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
                
            print(f"[Data Agent] ⚡ Extracting massively via native C unzip...")
            import subprocess
            subprocess.run(["unzip", "-q", "-o", str(extract_dir / filename), "-d", str(extract_dir)], check=True)
            
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


def train_lobert_worker(model, loader, max_steps=5000):
    import time
    print("\n[LOBERT-Thread] 🔥 Starting Continuous Fine-Tuning Steps...")
    step = 0
    while step < max_steps:
        for messages, timestamps, targets in loader:
            if step >= max_steps: break
            loss = model.finetune_step(messages, timestamps, targets)
            if step % 20 == 0:
                print(f"[LOBERT-Thread] Step {step:04d} | Loss: {loss:.4f}")
            step += 1
            time.sleep(0.01) # Small yield for the GIL
    print(f"\n[LOBERT-Thread] 🎉 Reached target steps!")

def train_fincast_worker(model, loader, max_steps=5000):
    import time
    print("\n[FinCast-Thread] 🔥 Starting Continuous Fine-Tuning Steps...")
    step = 0
    while step < max_steps:
        for x, y in loader:
            if step >= max_steps: break
            # FinCast finetune_step expects lists of numpy arrays
            ohlcv_windows = [win.numpy() for win in x]
            target_returns = [t.item() for t in y]
            loss = model.finetune_step(ohlcv_windows, target_returns)
            if step % 20 == 0:
                print(f"[FinCast-Thread] Step {step:04d} | Loss: {loss:.4f}")
            step += 1
            time.sleep(0.01) # Small yield for the GIL
    print(f"\n[FinCast-Thread] 🎉 Reached target steps!")

def run_online_adaptation_loop():
    print("\n🚀 Initializing Concurrent CPU Adaptation Loop...")
    import torch
    import numpy as np
    from src.core.lob_encoder import LOBERTModel
    from src.mission_control.forecast import FinCastForecaster
    from src.data.data_loaders import LOBERTDataset, FinCastDataset
    from torch.utils.data import DataLoader
    
    # ── CPU Thread Partitioning ──
    # PyTorch defaults to using all physical cores for math. We partition 
    # the threads so LOBERT and FinCast don't thrash each other context switching.
    num_cores = os.cpu_count() or 4
    torch.set_num_threads(max(1, num_cores // 2))
    print(f"⚙️ Hardware Optimization: Allocated {torch.get_num_threads()} CPU math threads per model.")

    # Initialize CPU models
    lobert = LOBERTModel()
    fincast = FinCastForecaster()

    lob_path = Path("data/lob_history.parquet")
    ohlcv_path = Path("data/ohlcv_history.parquet")

    # Verify data exists
    if not lob_path.exists() or not ohlcv_path.exists():
        print("❌ Data not found! Fetching data first...")
        return False

    print("\n📦 Spawning PyTorch DataLoaders with parallel background workers...")
    lob_dataset = LOBERTDataset(str(lob_path), seq_len=128)
    fincast_dataset = FinCastDataset(str(ohlcv_path), seq_len=512)

    # num_workers > 0 offloads parquet reading and tensor creation from the main GIL thread
    lob_loader = DataLoader(lob_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    fincast_loader = DataLoader(fincast_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    print("\n🚀 Launching Concurrent Execution Threads...")
    
    t_lobert = threading.Thread(target=train_lobert_worker, args=(lobert, lob_loader, 10000))
    t_fincast = threading.Thread(target=train_fincast_worker, args=(fincast, fincast_loader, 10000))
    
    t_lobert.start()
    t_fincast.start()
    
    t_lobert.join()
    t_fincast.join()

    print(f"\n🎉 Concurrent Adaptation Harness gracefully exited!")
    return True

def main():
    print("="*60)
    print("🤖 High-Efficiency CPU Online Adaptation Harness")
    print("="*60)

    try:
        repo_root = Path(__file__).resolve().parent.parent
        os.chdir(repo_root)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    except NameError:
        # User pasted the script directly into a Jupyter notebook cell
        if IN_COLAB and Path("/content/crypto").exists():
            os.chdir("/content/crypto")
            sys.path.append("/content/crypto")
    
    import shutil
    if IN_COLAB and (not shutil.which("aria2c") or not shutil.which("unzip")):
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
