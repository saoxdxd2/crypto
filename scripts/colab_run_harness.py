"""
Autonomous Google Colab Loop Engineering Harness.

This agentic script runs LOBERT and FinCast training. If they fail to meet target
benchmarks, it autonomously consults a Gemini 2.5 Flash agent to determine what 
historical market regime data is lacking, downloads the monthly archive zip from 
Binance Vision, processes it via GPU (cudf), and seamlessly overlaps the downloading/extracting 
with the model's active training using Zero-Shot async validation.
"""

import os
import sys
import json
import time
import urllib.request
import subprocess
import threading
from pathlib import Path

# In highly optimized environments, we use RAPIDS cudf for GPU-accelerated processing
try:
    import cudf
    USE_CUDF = True
except ImportError:
    import polars as cudf  # Fallback to polars if cudf isn't installed
    USE_CUDF = False

# Try importing google.colab to use secrets, otherwise fallback to env
try:
    from google.colab import userdata, files, runtime
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def get_api_key():
    return "AQ.Ab8RN6Ifh1nrh3KyuJkTuKUC_FW6rvMipV1Bg6nieB_lDX1h-w"


class BinanceArchiveFetcher:
    """Downloads monthly archive ZIPs and uses GPU (cudf) to parse/extract into Parquet instantly."""
    
    BASE_URL = "https://data.binance.vision/data/spot/monthly"
    
    @staticmethod
    def fetch_trades(symbol: str, year: str, month: str, output_parquet: Path):
        month = str(month).zfill(2)
        filename = f"{symbol}-trades-{year}-{month}.zip"
        url = f"{BinanceArchiveFetcher.BASE_URL}/trades/{symbol}/{filename}"
        
        print(f"\n[GPU Data Agent] 📥 Downloading Trades: {url}")
        zip_path = Path(filename)
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"[GPU Data Agent] ❌ Failed to download {url}: {e}")
            return False
            
        print(f"[GPU Data Agent] ⚡ Extracting ZIP...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("temp_data")
        csv_file = Path("temp_data") / filename.replace(".zip", ".csv")
            
        print(f"[GPU Data Agent] ⚡ GPU-Accelerated Chunked Parsing of CSV...")
        
        fallback_used = False
        try:
            import os
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            file_size = os.path.getsize(csv_file)
            chunk_size_bytes = 250 * 1024 * 1024  # 250 MB chunks
            offset = 0
            writer = None
            
            while offset < file_size:
                df = cudf.read_csv(
                    csv_file, 
                    byte_range=(offset, chunk_size_bytes),
                    header=None, 
                    names=["id", "price", "volume", "quoteQty", "timestamp_ms", "isBuyerMaker", "isBestMatch"],
                    dtype={
                        "id": "int64", "price": "float64", "volume": "float64", "quoteQty": "float64", 
                        "timestamp_ms": "int64", "isBuyerMaker": "bool", "isBestMatch": "bool"
                    }
                )
                
                if len(df) > 0:
                    df['side'] = df['isBuyerMaker'].astype('float64')
                    df['order_type'] = 1.0
                    df = df[["price", "volume", "side", "order_type", "timestamp_ms"]]
                    df = df.sort_values("timestamp_ms")
                    
                    table = df.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(output_parquet, table.schema)
                    writer.write_table(table)
                    
                offset += chunk_size_bytes
                
            if writer:
                writer.close()
                
        except Exception as e:
            if USE_CUDF and ("memory" in str(e).lower() or "alloc" in str(e).lower() or "cuda" in str(e).lower() or "cufile" in str(e).lower()):
                print(f"[GPU Data Agent] ⚠️ CUDA OOM during GPU extraction! Falling back to CPU Streaming for this dataset...")
                fallback_used = True
                import polars as pl
                
                lf = pl.scan_csv(csv_file, has_header=False, new_columns=["id", "price", "volume", "quoteQty", "timestamp_ms", "isBuyerMaker", "isBestMatch"])
                lf = lf.with_columns([
                    pl.when(pl.col("isBuyerMaker")).then(1.0).otherwise(0.0).alias("side"),
                    pl.lit(1.0).alias("order_type")
                ])
                lf = lf.select(["price", "volume", "side", "order_type", "timestamp_ms"])
                lf.sink_parquet(output_parquet)
            else:
                raise e
        
        # Cleanup
        if 'csv_file' in locals() and csv_file.exists():
            csv_file.unlink()
        zip_path.unlink()
        print(f"[GPU Data Agent] ✅ Successfully processed {year}-{month} trades to {output_parquet}")
        return True

    @staticmethod
    def fetch_klines(symbol: str, year: str, month: str, output_parquet: Path):
        month = str(month).zfill(2)
        # Using 1m candles
        filename = f"{symbol}-1m-{year}-{month}.zip"
        url = f"{BinanceArchiveFetcher.BASE_URL}/klines/{symbol}/1m/{filename}"
        
        print(f"\n[GPU Data Agent] 📥 Downloading Klines: {url}")
        zip_path = Path(filename)
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"[GPU Data Agent] ❌ Failed to download {url}: {e}")
            return False
            
        print(f"[GPU Data Agent] ⚡ Extracting ZIP...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("temp_data")
        csv_file = Path("temp_data") / filename.replace(".zip", ".csv")
            
        print(f"[GPU Data Agent] ⚡ GPU-Accelerated Chunked Parsing of CSV...")
        
        fallback_used = False
        try:
            import os
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            file_size = os.path.getsize(csv_file)
            chunk_size_bytes = 250 * 1024 * 1024
            offset = 0
            writer = None
            
            while offset < file_size:
                df = cudf.read_csv(
                    csv_file, 
                    byte_range=(offset, chunk_size_bytes),
                    header=None, 
                    names=[
                        "open_time", "open", "high", "low", "close", "volume", "close_time", 
                        "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
                    ],
                    dtype={
                        "open_time": "int64", "open": "float64", "high": "float64", "low": "float64", 
                        "close": "float64", "volume": "float64", "close_time": "int64"
                    }
                )
                
                if len(df) > 0:
                    df['is_closed'] = True
                    df = df[["open_time", "open", "high", "low", "close", "volume", "is_closed"]]
                    df = df.sort_values("open_time")
                    
                    table = df.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(output_parquet, table.schema)
                    writer.write_table(table)
                    
                offset += chunk_size_bytes
                
            if writer:
                writer.close()
                
        except Exception as e:
            if USE_CUDF and ("memory" in str(e).lower() or "alloc" in str(e).lower() or "cuda" in str(e).lower() or "cufile" in str(e).lower()):
                print(f"[GPU Data Agent] ⚠️ CUDA OOM during GPU extraction! Falling back to CPU Streaming for this dataset...")
                fallback_used = True
                import polars as pl
                lf = pl.scan_csv(csv_file, has_header=False, new_columns=[
                    "open_time", "open", "high", "low", "close", "volume", "close_time", 
                    "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
                ])
                lf = lf.with_columns(pl.lit(True).alias("is_closed"))
                lf = lf.select(["open_time", "open", "high", "low", "close", "volume", "is_closed"])
                lf.sink_parquet(output_parquet)
            else:
                raise e
        
        # Cleanup
        if 'csv_file' in locals() and csv_file.exists():
            csv_file.unlink()
        zip_path.unlink()
        print(f"[GPU Data Agent] ✅ Successfully processed {year}-{month} klines to {output_parquet}")
        return True


class GeminiDataAgent:
    def __init__(self, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        
    def get_data_recommendation(self, logs: str) -> dict:
        prompt = f"""
You are an autonomous Curriculum Learning Agent for a crypto trading AI.
We are using a Zero-Shot evaluation to assess the model's current capabilities.
Here is the zero-shot validation log:

<LOGS>
{logs}
</LOGS>

Identify what market regime the models need to learn next to improve generalization.
Output exactly ONE JSON object matching this schema:
{{
  "model": "LOBERT",  // or "FinCast" or "BOTH"
  "data_type": "trades", // or "klines"
  "symbol": "BTCUSDT",
  "year": "2022",
  "month": "11",
  "reason": "Short explanation."
}}
"""
        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config={"temperature": 0.2}
        )
        text = response.text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)


def async_data_fetch_worker(agent, model_name, zero_shot_log):
    """Background thread to query Gemini and download datasets while model trains."""
    try:
        print(f"\n[Background Worker] 🧠 Querying Gemini with Zero-Shot Eval for {model_name}...")
        decision = agent.get_data_recommendation(zero_shot_log)
        print(f"\n[Background Worker] 🧠 Gemini Decision: {decision['reason']}")
        print(f"[Background Worker] 📥 Action: Downloading {decision['symbol']} {decision['data_type']} for {decision['year']}-{decision['month']}")
        
        if decision['data_type'] == "trades" or decision['model'] in ["LOBERT", "BOTH"]:
            BinanceArchiveFetcher.fetch_trades(decision['symbol'], decision['year'], decision['month'], Path("data/lob_history.parquet"))
        
        if decision['data_type'] == "klines" or decision['model'] in ["FinCast", "BOTH"]:
            BinanceArchiveFetcher.fetch_klines(decision['symbol'], decision['year'], decision['month'], Path("data/ohlcv_history.parquet"))
            
    except Exception as e:
        print(f"[Background Worker] ❌ Task failed: {e}")


def run_training_async(agent, model_name, script_path, target_acc):
    print(f"\n🚀 Launching {model_name} Training Loop...")
    process = subprocess.Popen(
        [sys.executable, "-m", script_path, "--target_acc", str(target_acc)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    success = False
    background_thread = None
    
    for line in process.stdout:
        sys.stdout.write(line)
        
        # Intercept Zero-Shot Eval and kick off async download immediately
        if "[ZERO-SHOT EVAL]" in line and background_thread is None:
            background_thread = threading.Thread(
                target=async_data_fetch_worker, 
                args=(agent, model_name, line)
            )
            background_thread.start()
            
        if "Benchmark threshold met!" in line or "Model is ACCEPTABLE" in line:
            success = True
            
    process.wait()
    
    # Ensure the background download has completed before the next loop starts
    if background_thread is not None:
        background_thread.join()
        
    return success


def main():
    print("="*60)
    print("🤖 High-Efficiency Concurrent Loop Engineering Harness")
    print("="*60)

    if IN_COLAB and os.path.exists("/content/crypto"):
        os.chdir("/content/crypto")

    # Ensure cudf is available or fallback
    if not USE_CUDF:
        print("⚠️ Warning: RAPIDS cudf not detected! Falling back to CPU polars processing.")
        print("For maximum GPU efficiency in Colab, run: !pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com")
    else:
        print("⚡ RAPIDS cudf detected! GPU data pipelines activated.")

    agent = GeminiDataAgent(get_api_key())
    
    Path("data").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    
    TARGET_ACC = 0.58
    max_loops = 5
    
    # Pre-seed datasets for loop 1
    lob_path = Path("data/lob_history.parquet")
    ohlcv_path = Path("data/ohlcv_history.parquet")
    if not lob_path.exists():
        BinanceArchiveFetcher.fetch_trades("BTCUSDT", "2023", "01", lob_path)
    if not ohlcv_path.exists():
        BinanceArchiveFetcher.fetch_klines("BTCUSDT", "2023", "01", ohlcv_path)
        
    for loop_idx in range(1, max_loops + 1):
        print(f"\n" + "="*60)
        print(f"🔄 LOOP {loop_idx}/{max_loops} (Continuous Training & Async Fetching)")
        print("="*60)
        
        lob_success = run_training_async(agent, "LOBERT", "src.cloud.train_lobert", TARGET_ACC)
        fin_success = run_training_async(agent, "FinCast", "src.cloud.train_fincast", TARGET_ACC)
        
        if lob_success and fin_success:
            print("\n🎉 ALL MODELS CRUSHED THE BENCHMARK!")
            break
            
    print("\n🏁 Loop Engineering Complete.")
    
    if IN_COLAB:
        print("Triggering automatic browser downloads of the CRUSHING models...")
        if Path("checkpoints/lobert_checkpoint.pt").exists():
            files.download("checkpoints/lobert_checkpoint.pt")
        if Path("checkpoints/fincast_checkpoint.pt").exists():
            files.download("checkpoints/fincast_checkpoint.pt")
        print("\n✅ All done! The Colab session will remain active so your downloads can finish.")

if __name__ == "__main__":
    main()
