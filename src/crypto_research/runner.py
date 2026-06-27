import multiprocessing
import subprocess
import sys
import logging
from pathlib import Path
from crypto_research.gui.app import CyberQuantDashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_flaml_optimizer():
    """Background process that runs FLAML optimization daily."""
    import time
    from crypto_research.flaml_optimizer import FlamlOptimizer
    
    logger.info("Starting FLAML Optimizer daemon...")
    optimizer = FlamlOptimizer()
    
    while True:
        try:
            logger.info("Running FLAML tuning pass...")
            optimizer.optimize_thresholds(time_budget=10) # 10 seconds for demo
            time.sleep(3600) # Sleep for 1 hour
        except Exception as e:
            logger.error(f"FLAML optimizer crashed: {e}")
            time.sleep(60)

def start_freqtrade_docker():
    """Starts the Freqtrade execution engine via docker-compose."""
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        logger.info("Freqtrade Docker started.")
    except Exception as e:
        logger.error(f"Failed to start Docker: {e}")
        logger.warning("Is Docker Desktop running?")

if __name__ == "__main__":
    # Required for multiprocessing in PyInstaller executables
    multiprocessing.freeze_support()
    
    logger.info("Initializing Mission Control Orchestrator...")
    
    # Ensure data directories exist
    Path("data/signals").mkdir(parents=True, exist_ok=True)
    Path("data/metadata").mkdir(parents=True, exist_ok=True)
    
    # 1. Start Freqtrade Docker in the background
    start_freqtrade_docker()
    
    # 2. Start FLAML Optimizer as a separate process
    flaml_proc = multiprocessing.Process(target=start_flaml_optimizer, daemon=True)
    flaml_proc.start()
    
    # 3. Launch the CustomTkinter GUI on the main thread
    try:
        logger.info("Launching GUI...")
        app = CyberQuantDashboard()
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Cleanup
        if flaml_proc.is_alive():
            flaml_proc.terminate()
        logger.info("Mission Control shutdown complete.")
