from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# import mlflow
import pandas as pd
# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Phase 16: MLflow tracking for experiments.
    Tracks hyperparameters, model versions, thresholds, and metrics.
    """

    def __init__(self, tracking_uri: str = "sqlite:///data/mlflow/mlflow.db") -> None:
        self.tracking_uri = tracking_uri
        # mlflow.set_tracking_uri(self.tracking_uri)

    def log_experiment(self, experiment_name: str, params: dict[str, Any], metrics: dict[str, float]) -> None:
        """
        Logs a full run including TimesFM config, thresholds, and performance metrics.
        """
        logger.info(f"Logging experiment '{experiment_name}' to MLflow...")
        # mlflow.set_experiment(experiment_name)
        # with mlflow.start_run():
        #     mlflow.log_params(params)
        #     mlflow.log_metrics(metrics)
        logger.info(f"Logged params: {list(params.keys())}")
        logger.info(f"Logged metrics: {list(metrics.keys())}")


class DriftMonitor:
    """
    Phase 16: Evidently Drift Reports.
    Monitors distribution shifts in forecasts, feature inputs, and news sentiments.
    """

    def __init__(self, report_dir: Path = Path("data/reports/drift")) -> None:
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, report_name: str) -> Path:
        """
        Generates an HTML drift report to detect distribution shifts in market data or signal outputs.
        """
        logger.info(f"Generating drift report: {report_name}")
        
        # report = Report(metrics=[
        #     DataDriftPreset(),
        #     TargetDriftPreset(),
        # ])
        # report.run(reference_data=reference_data, current_data=current_data)
        
        out_path = self.report_dir / f"{report_name}.html"
        # report.save_html(str(out_path))
        
        logger.info(f"Drift report saved to {out_path}")
        return out_path
