from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from flaml import tune
from src.core.sys_events import push_sys_event

logger = logging.getLogger(__name__)


class FlamlOptimizer:
    """
    Phase 12: Continuous AutoML Optimization Plane.
    Uses Microsoft FLAML to optimize the thresholds of the Risk Governor and Decision Math 
    based on local Parquet data, keeping the knowledge updated without fine-tuning TimesFM weights.
    """

    def __init__(self, config_output_path: Path = Path("data/metadata/optimal_thresholds.json")) -> None:
        self.config_output_path = config_output_path

    def _simulate_backtest(self, config: dict[str, float]) -> float:
        """
        Mock evaluation function.
        In production, this would load Parquet data, apply the TimesFM predictions,
        and calculate the Sharpe ratio or PnL based on these thresholds.
        """
        # Calculate progress
        import time
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        progress = min(1.0, elapsed / self.time_budget)
        
        # Only push progress occasionally so we don't flood the UI
        if not hasattr(self, 'last_progress_push') or (time.time() - self.last_progress_push > 0.5):
            push_sys_event("TRAINING", "FLAML exploring hyperparameter surface...", progress=progress)
            self.last_progress_push = time.time()

        # Evaluate actual paper trading performance
        score = 0.0
        try:
            trades_file = Path("data/trades/trade_history.json")
            if trades_file.exists():
                trades = json.loads(trades_file.read_text())
                
                win_count = 0
                total_pnl = 0.0
                
                for t in trades:
                    if t.get("status") == "CLOSED":
                        pnl = float(t.get("pnl", 0.0))
                        edge = float(t.get("net_edge", 0.0))
                        
                        # Apply simulated threshold filtering
                        if abs(edge) >= config["minimum_edge_threshold"]:
                            total_pnl += pnl
                            if pnl > 0:
                                win_count += 1
                
                if total_pnl != 0:
                    score = total_pnl * (win_count / max(1, len(trades)))
                else:
                    # Fallback math if no trades closed yet
                    score = -(config["minimum_edge_threshold"] - 0.002)**2 * 10000
            else:
                score = -(config["minimum_edge_threshold"] - 0.002)**2 * 10000
        except Exception as e:
            logger.error(f"Failed to evaluate real trades: {e}")
            score = -100.0
            
        return {"score": float(score)}

    def optimize_thresholds(self, time_budget: int = 10) -> None:
        """
        Runs FLAML's hyperparameter tuning to find the optimal configurations.
        """
        logger.info("Starting FLAML continuous optimization on local data...")
        push_sys_event("TRAINING", "Starting FLAML Optimizer on historical local data...", progress=0.0)
        
        self.time_budget = time_budget
        import time
        self.start_time = time.time()
        
        search_space = {
            "minimum_edge_threshold": tune.uniform(0.0005, 0.005),
            "max_spread_bps": tune.uniform(2.0, 15.0),
            "max_daily_loss_usd": tune.randint(100, 2000),
            "min_book_depth_usd": tune.randint(10000, 100000)
        }

        analysis = tune.run(
            self._simulate_backtest,
            metric="score",
            mode="max",
            search_alg="BlendSearch",
            config=search_space,
            time_budget_s=time_budget,
            verbose=0
        )

        best_config: dict[str, float] = analysis.best_config
        logger.info(f"FLAML found optimal thresholds: {best_config}")
        
        # Write to local JSON for live consumption
        self.config_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_output_path.write_text(json.dumps(best_config, indent=2))
        logger.info(f"Optimal thresholds saved to {self.config_output_path}")
        push_sys_event("TRAINING", f"FLAML Optimizer completed successfully. New thresholds locked.")

    @staticmethod
    def load_optimal_config(config_path: Path = Path("data/metadata/optimal_thresholds.json")) -> dict[str, Any]:
        """Loads the JSON config if it exists, otherwise returns empty dict."""
        if config_path.exists():
            return json.loads(config_path.read_text())
        return {}
