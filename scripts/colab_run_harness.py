"""
Autonomous Google Colab Loop Engineering Harness.

This agentic script runs LOBERT and FinCast training. If they fail to meet target
benchmarks, it autonomously consults a Gemini 2.5 Flash agent to determine what 
historical market regime data is lacking, downloads the monthly archive zip from 
Binance Vision, processes it, and restarts the training loop.
"""

import os
import sys
import json
import time
import zipfile
import urllib.request
import subprocess
from pathlib import Path
import polars as pl

# Try importing google.colab to use secrets, otherwise fallback to env
try:
    from google.colab import userdata, files, runtime
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def get_api_key():
    if IN_COLAB:
        try:
            return userdata.get('GEMINI_API_KEY')
        except:
            return os.environ.get("GEMINI_API_KEY")
    return os.environ.get("GEMINI_API_KEY")


class BinanceArchiveFetcher:
    """Downloads monthly archive ZIPs from data.binance.vision and converts to required Parquet."""
    
    BASE_URL = "https://data.binance.vision/data/spot/monthly"
    
    @staticmethod
    def fetch_trades(symbol: str, year: str, month: str, output_parquet: Path):
        month = str(month).zfill(2)
        filename = f"{symbol}-trades-{year}-{month}.zip"
        url = f"{BinanceArchiveFetcher.BASE_URL}/trades/{symbol}/{filename}"
        
        print(f"\n[Data Agent] Downloading Trades: {url}")
        zip_path = Path(filename)
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"[Data Agent] ❌ Failed to download {url}: {e}")
            return False
            
        print(f"[Data Agent] Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("temp_data")
            
        csv_file = Path("temp_data") / filename.replace(".zip", ".csv")
        
        # Schema: id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
        print(f"[Data Agent] Processing CSV into Parquet...")
        df = pl.read_csv(csv_file, has_header=False, new_columns=["id", "price", "volume", "quoteQty", "timestamp_ms", "isBuyerMaker", "isBestMatch"])
        
        # Convert to LOBERT schema: price, volume, side, order_type, timestamp_ms
        df = df.with_columns([
            pl.col("price").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.when(pl.col("isBuyerMaker")).then(1.0).otherwise(0.0).alias("side"),
            pl.lit(1.0).alias("order_type"),
            pl.col("timestamp_ms").cast(pl.Int64)
        ]).select(["price", "volume", "side", "order_type", "timestamp_ms"])
        
        df = df.sort("timestamp_ms")
        df.write_parquet(output_parquet)
        
        # Cleanup
        zip_path.unlink()
        csv_file.unlink()
        print(f"[Data Agent] ✅ Successfully appended {year}-{month} trades to {output_parquet}")
        return True

    @staticmethod
    def fetch_klines(symbol: str, year: str, month: str, output_parquet: Path):
        month = str(month).zfill(2)
        # Using 1m candles
        filename = f"{symbol}-1m-{year}-{month}.zip"
        url = f"{BinanceArchiveFetcher.BASE_URL}/klines/{symbol}/1m/{filename}"
        
        print(f"\n[Data Agent] Downloading Klines: {url}")
        zip_path = Path(filename)
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"[Data Agent] ❌ Failed to download {url}: {e}")
            return False
            
        print(f"[Data Agent] Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("temp_data")
            
        csv_file = Path("temp_data") / filename.replace(".zip", ".csv")
        
        # Schema: open_time, open, high, low, close, volume, close_time, quote_asset_volume, trades, taker_buy_base, taker_buy_quote, ignore
        print(f"[Data Agent] Processing CSV into Parquet...")
        df = pl.read_csv(csv_file, has_header=False, new_columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time", 
            "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        
        # Convert to FinCast schema: open, high, low, close, volume, is_closed
        df = df.with_columns([
            pl.col("open_time").cast(pl.Int64),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.lit(True).alias("is_closed")
        ]).select(["open", "high", "low", "close", "volume", "is_closed"])
        
        df = df.sort("open_time")
        df.write_parquet(output_parquet)
        
        # Cleanup
        zip_path.unlink()
        csv_file.unlink()
        print(f"[Data Agent] ✅ Successfully appended {year}-{month} klines to {output_parquet}")
        return True


class GeminiDataAgent:
    def __init__(self, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        
    def get_data_recommendation(self, logs: str) -> dict:
        prompt = f"""
You are an autonomous Curriculum Learning Agent for a crypto trading AI.
The models (LOBERT for tick data, FinCast for kline data) just finished a training loop.
Here are their validation logs showing where they failed to meet the target benchmark:

<LOGS>
{logs}
</LOGS>

Your job is to identify what market regime the models need to learn next to improve their generalization, and instruct the data fetcher to download it from the Binance Vision monthly archives.

Examples of regimes:
- High volatility / bear market crashes (e.g. 2022-11 FTX crash, 2022-05 LUNA crash)
- Extreme bull runs (e.g. 2021-02, 2021-10)
- Sideways / low volatility (e.g. 2023-08)

Output exactly ONE JSON object with no markdown formatting. It MUST match this schema:
{{
  "model": "LOBERT",  // or "FinCast" or "BOTH"
  "data_type": "trades", // or "klines"
  "symbol": "BTCUSDT",
  "year": "2022",
  "month": "11",
  "reason": "Short explanation of why this specific month is needed."
}}
"""
        print("[Gemini Agent] Analyzing logs and querying LLM for curriculum advice...")
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"temperature": 0.2}
        )
        
        text = response.text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
            
        return json.loads(text)


def run_training(model_name, script_path, target_acc):
    print(f"\n🚀 Launching {model_name} Training Loop...")
    process = subprocess.Popen(
        [sys.executable, "-m", script_path, "--target_acc", str(target_acc)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    logs = []
    success = False
    for line in process.stdout:
        sys.stdout.write(line)
        logs.append(line)
        if "Benchmark threshold met!" in line or "Model is ACCEPTABLE" in line:
            success = True
            
    process.wait()
    return success, "".join(logs[-30:])  # Return last 30 lines for the LLM context


def main():
    print("="*60)
    print("🤖 Autonomous Loop Engineering Harness")
    print("="*60)

    # In Colab, force working directory to repo root
    if IN_COLAB and os.path.exists("/content/crypto"):
        os.chdir("/content/crypto")

    # Install requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "google-genai", "polars", "pyarrow", "huggingface_hub"], check=True)

    api_key = get_api_key()
    if not api_key:
        print("❌ CRITICAL ERROR: GEMINI_API_KEY is not set!")
        if IN_COLAB:
            print("Please add 'GEMINI_API_KEY' to your Colab Secrets (the key icon on the left panel) and toggle 'Notebook access'.")
        sys.exit(1)

    agent = GeminiDataAgent(api_key)
    
    Path("data").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("temp_data").mkdir(exist_ok=True)
    
    TARGET_ACC = 0.58
    max_loops = 5
    
    # Pre-seed datasets if they are completely empty (so the first training loop doesn't crash)
    lob_path = Path("data/lob_history.parquet")
    ohlcv_path = Path("data/ohlcv_history.parquet")
    if not lob_path.exists():
        BinanceArchiveFetcher.fetch_trades("BTCUSDT", "2023", "01", lob_path)
    if not ohlcv_path.exists():
        BinanceArchiveFetcher.fetch_klines("BTCUSDT", "2023", "01", ohlcv_path)
        
    for loop_idx in range(1, max_loops + 1):
        print(f"\n" + "="*60)
        print(f"🔄 LOOP ENGINEERING ITERATION {loop_idx}/{max_loops}")
        print("="*60)
        
        lob_success, lob_logs = run_training("LOBERT", "src.cloud.train_lobert", TARGET_ACC)
        fin_success, fin_logs = run_training("FinCast", "src.cloud.train_fincast", TARGET_ACC)
        
        if lob_success and fin_success:
            print("\n🎉 ALL MODELS CRUSHED THE BENCHMARK!")
            break
            
        print("\n⚠️ Models failed to meet target accuracy. Engaging Gemini Data Agent...")
        combined_logs = f"--- LOBERT LOGS ---\n{lob_logs}\n\n--- FINCAST LOGS ---\n{fin_logs}"
        
        try:
            decision = agent.get_data_recommendation(combined_logs)
            print(f"\n🧠 Gemini Decision: {decision['reason']}")
            print(f"📥 Action: Downloading {decision['symbol']} {decision['data_type']} for {decision['year']}-{decision['month']}")
            
            # Fetch the data requested by the LLM
            if decision['data_type'] == "trades" or decision['model'] in ["LOBERT", "BOTH"]:
                BinanceArchiveFetcher.fetch_trades(decision['symbol'], decision['year'], decision['month'], lob_path)
            
            if decision['data_type'] == "klines" or decision['model'] in ["FinCast", "BOTH"]:
                BinanceArchiveFetcher.fetch_klines(decision['symbol'], decision['year'], decision['month'], ohlcv_path)
                
        except Exception as e:
            print(f"❌ Gemini Agent Loop failed: {e}. Will retry in next loop.")
            time.sleep(5)
            continue
            
    print("\n" + "="*60)
    print("🏁 Loop Engineering Complete.")
    
    if IN_COLAB:
        print("Triggering automatic browser downloads of the CRUSHING models...")
        if Path("checkpoints/lobert_checkpoint.pt").exists():
            files.download("checkpoints/lobert_checkpoint.pt")
        if Path("checkpoints/fincast_checkpoint.pt").exists():
            files.download("checkpoints/fincast_checkpoint.pt")
            
        print("Shutting down Colab instance to save credits...")
        runtime.unassign()

if __name__ == "__main__":
    main()
