import multiprocessing
import subprocess
from pathlib import Path
import sys
import os

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core.logger import logger

# ── Multi-Coin Configuration ──
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def start_flaml_optimizer():
    """Background process that runs FLAML optimization hourly."""
    import time
    from src.mission_control.flaml_optimizer import FlamlOptimizer

    logger.info("Starting FLAML Optimizer daemon...")
    optimizer = FlamlOptimizer()

    while True:
        try:
            logger.info("Running FLAML tuning pass...")
            optimizer.optimize_thresholds(time_budget=10)
            time.sleep(3600)
        except Exception as e:
            logger.error(f"FLAML optimizer crashed: {e}")
            time.sleep(60)

def start_reasoning_daemon():
    """Background process that runs the Deep Reasoning Loop Engineering."""
    import time
    from src.mission_control.deep_reasoning import DeepReasoningSpecialist
    
    logger.info("Starting Deep Reasoning Daemon...")
    # This instantiation will download the model if missing and log it to the UI
    dr = DeepReasoningSpecialist()
    
    # Run a mock loop engineering cycle every 60 seconds to demonstrate to the user
    while True:
        try:
            logger.info("Running Deep Reasoning evaluation pass...")
            
            # Calculate current portfolio balance
            balance = 10000.0
            import json
            from pathlib import Path
            trades_path = Path("data/trades/trade_history.json")
            if trades_path.exists():
                try:
                    trades = json.loads(trades_path.read_text(encoding="utf-8"))
                    for t in trades:
                        balance += float(t.get("pnl", 0))
                except:
                    pass
            
            # Load active settings
            settings = {}
            settings_path = Path("data/metadata/optimal_thresholds.json")
            if settings_path.exists():
                try:
                    settings = json.loads(settings_path.read_text(encoding="utf-8"))
                except:
                    pass
            
            # Load latest signal
            sig_edge = 0.0
            sig_pattern = 0.5
            sig_sym = "BTCUSDT"
            
            sig_path = Path("data/signals/LOBERT_1m.json")
            if sig_path.exists():
                try:
                    sigs = json.loads(sig_path.read_text(encoding="utf-8"))
                    if sigs:
                        last_sig = sigs[-1]
                        sig_sym = last_sig.get("symbol", "BTCUSDT")
                        sig_edge = float(last_sig.get("book_imbalance", 0.0))
                        sig_pattern = float(last_sig.get("confidence", 0.5))
                except Exception as e:
                    logger.error(f"Error parsing signals: {e}")
            
            dr.analyze_market_regime(
                symbol=sig_sym,
                timesfm_edge=sig_edge,
                pattern_score=sig_pattern,
                news_summary="Live paper trading active.",
                portfolio_balance=balance,
                user_settings=settings
            )
            time.sleep(60)
        except Exception as e:
            logger.error(f"Deep Reasoning daemon crashed: {e}")
            time.sleep(60)


def start_freqtrade_docker():
    """Starts the Freqtrade execution engine via docker-compose."""
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        logger.info("Freqtrade Docker started.")
    except Exception as e:
        logger.error(f"Failed to start Docker: {e}")
        logger.warning("Is Docker Desktop running? Continuing without Freqtrade...")

def start_llm_server():
    from src.mission_control.llm_server import run_server
    run_server(port=5001)

def start_paper_trader():
    from src.mission_control.live_paper_trader import run_paper_trader
    run_paper_trader()


if __name__ == "__main__":
    # Required for multiprocessing in PyInstaller executables
    multiprocessing.freeze_support()

    logger.info("Initializing Mission Control Orchestrator...")
    logger.info(f"Symbols: {SYMBOLS}")

    # Ensure data directories exist for all symbols
    for sym in SYMBOLS:
        Path(f"data/signals/{sym}").mkdir(parents=True, exist_ok=True)
    Path("data/signals").mkdir(parents=True, exist_ok=True)
    Path("data/metadata").mkdir(parents=True, exist_ok=True)
    Path("data/trades").mkdir(parents=True, exist_ok=True)
    Path("data/rl").mkdir(parents=True, exist_ok=True)

    # Initial run cleanup
    if not (Path("data/trades/trade_history.json")).exists():
        logger.info("First run detected — starting with clean slate...")

    # 1. Start Freqtrade Docker in the background
    start_freqtrade_docker()

    # 1.5 Start LLM Server
    llm_proc = multiprocessing.Process(target=start_llm_server, daemon=True)
    llm_proc.start()

    # 2. Start FLAML Optimizer as a separate process
    flaml_proc = multiprocessing.Process(target=start_flaml_optimizer, daemon=True)
    flaml_proc.start()

    # 2.5 Start Deep Reasoning Daemon
    dr_proc = multiprocessing.Process(target=start_reasoning_daemon, daemon=True)
    dr_proc.start()

    # 2.6 Start Live Paper Trader
    trader_proc = multiprocessing.Process(target=start_paper_trader, daemon=True)
    trader_proc.start()

    # 3. Launch the CustomTkinter GUI on the main thread
    try:
        logger.info("Launching GUI...")
        from src.mission_control.gui.app import CyberQuantDashboard
        app = CyberQuantDashboard()
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if flaml_proc.is_alive():
            flaml_proc.terminate()
        if dr_proc.is_alive():
            dr_proc.terminate()
        if llm_proc.is_alive():
            llm_proc.terminate()
        if trader_proc.is_alive():
            trader_proc.terminate()
        logger.info("Mission Control shutdown complete.")
