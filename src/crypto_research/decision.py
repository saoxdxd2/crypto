from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from crypto_research.flaml_optimizer import FlamlOptimizer

logger = logging.getLogger(__name__)


class DeterministicDecisionMath:
    """
    Phase 5: The pure deterministic math layer that evaluates TimesFM forecasts and News risk modifiers.
    Must never execute LLM logic directly.
    """

    def __init__(
        self,
        fee_rate_entry: float = 0.0005, # Example 5 bps
        fee_rate_exit: float = 0.0005,
        spread_estimate: float = 0.0001, # Example 1 bps
        slippage_estimate: float = 0.0002, # Example 2 bps
        minimum_edge_threshold: float = 0.0015,
        max_signal_age_seconds: int = 60,
        max_position_risk: float = 0.004,
    ) -> None:
        self.fee_rate_entry = fee_rate_entry
        self.fee_rate_exit = fee_rate_exit
        self.spread_estimate = spread_estimate
        self.slippage_estimate = slippage_estimate
        self.minimum_edge_threshold = minimum_edge_threshold
        self.max_signal_age_seconds = max_signal_age_seconds
        self.max_position_risk = max_position_risk

        # Phase 12: Load FLAML Optimized Thresholds
        optimized_config = FlamlOptimizer.load_optimal_config()
        if "minimum_edge_threshold" in optimized_config:
            self.minimum_edge_threshold = optimized_config["minimum_edge_threshold"]
            logger.info(f"Loaded FLAML optimized minimum_edge_threshold: {self.minimum_edge_threshold}")

    def evaluate(
        self,
        forecast: dict[str, object],
        news_event: dict[str, object] | None,
        current_time: datetime | None = None,
    ) -> dict[str, object]:
        """
        Computes the net edge and strictly formats the signal JSON.
        """
        current_time = current_time or datetime.now(UTC)
        forecast_time = datetime.fromisoformat(str(forecast["created_at"]))
        
        # 1. Age Check
        age_seconds = (current_time - forecast_time).total_seconds()
        if age_seconds > self.max_signal_age_seconds:
            return self._build_signal(forecast, "hold", "none", 0.0, "SIGNAL_EXPIRED")

        # 2. Extract Forecast Logic
        expected_return = float(forecast["expected_return"])  # type: ignore
        p10 = float(forecast["p10_return"])  # type: ignore
        p90 = float(forecast["p90_return"])  # type: ignore
        
        # Compute absolute spread between quantiles as an uncertainty penalty
        uncertainty_penalty = abs(p90 - p10) * 0.1 # e.g. 10% of the prediction spread

        # 3. Apply News Risk Modifier
        risk_modifier = 1.0
        if news_event:
            news_time = datetime.fromisoformat(str(news_event["created_at"]))
            news_age = (current_time - news_time).total_seconds()
            if news_age <= 3600: # Only apply if news is recent (e.g. < 1 hour)
                risk_modifier = float(news_event["risk_modifier"])  # type: ignore

        adjusted_return = expected_return * risk_modifier

        # 4. Compute Net Edge
        estimated_cost = self.fee_rate_entry + self.fee_rate_exit + self.spread_estimate + self.slippage_estimate
        
        net_edge = adjusted_return - estimated_cost - uncertainty_penalty

        # 5. Determine Action
        if net_edge >= self.minimum_edge_threshold:
            action = "open"
            side = "long"
            reason_code = "TIMESFM_EDGE_NEWS_SAFE_COST_OK"
        elif net_edge <= -self.minimum_edge_threshold:
            # If shorting is allowed. The user said v1 is spot, so usually no short.
            # But the schema allows short. Let's just emit hold for now since Binance spot shorting requires margin.
            # We'll emit close if we had a position, but since this is stateless, the signal just says open/short.
            # Let's keep it long only for now as it's spot.
            action = "hold"
            side = "none"
            reason_code = "NEGATIVE_EDGE_HOLDING"
        else:
            action = "hold"
            side = "none"
            reason_code = "INSUFFICIENT_EDGE"

        # Apply position sizing
        size = 0.0
        if action == "open":
            size = self.max_position_risk / max(abs(p10), 0.0001) # Example Kelly/risk sizing
            size = min(size, 1.0) # Max 100% of capital
            
            # If risk_modifier is very low, reduce size
            if risk_modifier < 1.0:
                size *= risk_modifier

        return self._build_signal(forecast, action, side, net_edge, reason_code, size)

    def _build_signal(
        self,
        forecast: dict[str, object],
        action: str,
        side: str,
        net_edge: float,
        reason_code: str,
        size: float = 0.0,
    ) -> dict[str, object]:
        expires_at = datetime.fromisoformat(str(forecast["created_at"])) + timedelta(seconds=self.max_signal_age_seconds)
        
        return {
            "signal_id": f"sig_{uuid.uuid4().hex[:8]}",
            "created_at": datetime.now(UTC).isoformat(),
            "expires_at": expires_at.isoformat(),
            "exchange": forecast["exchange"],
            "symbol": forecast["symbol"],
            "action": action,
            "side": side,
            "size": round(size, 4),
            "max_loss": self.max_position_risk,
            "net_edge": round(net_edge, 6),
            "reason_code": reason_code,
        }

    def write_signal(self, signal: dict[str, object], output_dir: Path) -> Path:
        """Writes the signal payload cleanly."""
        output_dir.mkdir(parents=True, exist_ok=True)
        # Often Freqtrade will read a `latest_signal.json`
        out_path = output_dir / "latest_signal.json"
        out_path.write_text(json.dumps(signal, indent=2), encoding="utf-8")
        return out_path
