from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from flaml import tune

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
        # Example penalty if thresholds are too loose
        penalty = 0.0
        if config["minimum_edge_threshold"] < 0.001:
            penalty += 10.0
        
        # We want to maximize this score (e.g. Sharpe Ratio)
        # Mocking a convex optimization surface
        score = -(
            (config["minimum_edge_threshold"] - 0.002)**2 * 10000 + 
            (config["max_spread_bps"] - 5.0)**2
        ) - penalty
        
        return {"score": float(score)}

    def optimize_thresholds(self, time_budget: int = 10) -> None:
        """
        Runs FLAML's hyperparameter tuning to find the optimal configurations.
        """
        logger.info("Starting FLAML continuous optimization on local data...")
        
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

    @staticmethod
    def load_optimal_config(config_path: Path = Path("data/metadata/optimal_thresholds.json")) -> dict[str, Any]:
        """Loads the JSON config if it exists, otherwise returns empty dict."""
        if config_path.exists():
            return json.loads(config_path.read_text())
        return {}
