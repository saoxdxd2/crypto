from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from crypto_research.flaml_optimizer import FlamlOptimizer

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    spread_bps: float
    book_depth_usd: float
    is_api_stable: bool
    data_age_seconds: float


@dataclass
class PortfolioState:
    daily_loss_usd: float
    open_trades_count: int
    position_mismatch: bool


class RiskGovernor:
    """
    Phase 11 & Phase 9: Risk Governor and Kill Switches
    Deterministic and final gatekeeper before execution.
    Returns strictly: ALLOW, REDUCE, or BLOCK.
    """

    def __init__(
        self,
        max_spread_bps: float = 8.0,
        min_book_depth_usd: float = 50000.0,
        max_daily_loss_usd: float = 1000.0,
        max_open_trades: int = 3,
        max_signal_age_seconds: int = 60,
        max_data_age_seconds: int = 15,
        kill_switch_active: bool = False,
    ) -> None:
        self.max_spread_bps = max_spread_bps
        self.min_book_depth_usd = min_book_depth_usd
        self.max_daily_loss_usd = max_daily_loss_usd
        self.max_open_trades = max_open_trades
        self.max_signal_age_seconds = max_signal_age_seconds
        self.max_data_age_seconds = max_data_age_seconds
        self.kill_switch_active = kill_switch_active
        self.processed_signals: set[str] = set()

        # Phase 12: Load FLAML Optimized Thresholds
        optimized_config = FlamlOptimizer.load_optimal_config()
        if "max_spread_bps" in optimized_config:
            self.max_spread_bps = optimized_config["max_spread_bps"]
            logger.info(f"Loaded FLAML optimized max_spread_bps: {self.max_spread_bps}")
        if "min_book_depth_usd" in optimized_config:
            self.min_book_depth_usd = optimized_config["min_book_depth_usd"]
            logger.info(f"Loaded FLAML optimized min_book_depth_usd: {self.min_book_depth_usd}")
        if "max_daily_loss_usd" in optimized_config:
            self.max_daily_loss_usd = optimized_config["max_daily_loss_usd"]
            logger.info(f"Loaded FLAML optimized max_daily_loss_usd: {self.max_daily_loss_usd}")

    def evaluate(
        self,
        signal: dict[str, object],
        market: MarketState,
        portfolio: PortfolioState,
        news_confidence: float = 1.0,
        is_major_event: bool = False,
    ) -> str:
        """
        Evaluates all conditions and returns ALLOW, REDUCE, or BLOCK.
        """
        signal_id = str(signal["signal_id"])

        # 1. Kill Switch
        if self.kill_switch_active:
            logger.critical("BLOCK: Kill switch is ACTIVE.")
            return "BLOCK"

        # 2. Duplicate Signal
        if signal_id in self.processed_signals:
            logger.warning(f"BLOCK: Duplicate signal {signal_id}.")
            return "BLOCK"

        # 3. Expiration & Stale Data
        now = datetime.now(UTC)
        expires_at = datetime.fromisoformat(str(signal["expires_at"]))
        if now > expires_at:
            logger.warning("BLOCK: Signal expired.")
            return "BLOCK"
            
        if market.data_age_seconds > self.max_data_age_seconds:
            logger.warning(f"BLOCK: Market data stale ({market.data_age_seconds}s).")
            return "BLOCK"

        # 4. Exchange & Microstructure Health
        if not market.is_api_stable:
            logger.critical("BLOCK: Exchange API is unstable.")
            return "BLOCK"
            
        if market.spread_bps > self.max_spread_bps:
            logger.warning(f"BLOCK: Spread too wide ({market.spread_bps} bps).")
            return "BLOCK"
            
        if market.book_depth_usd < self.min_book_depth_usd:
            logger.warning(f"BLOCK: Insufficient book depth (${market.book_depth_usd}).")
            return "BLOCK"

        # 5. Portfolio & Risk Limits
        if portfolio.daily_loss_usd >= self.max_daily_loss_usd:
            logger.critical(f"BLOCK: Daily loss limit reached (${portfolio.daily_loss_usd}).")
            return "BLOCK"
            
        if portfolio.open_trades_count >= self.max_open_trades:
            logger.warning("BLOCK: Max open trades reached.")
            return "BLOCK"
            
        if portfolio.position_mismatch:
            logger.critical("BLOCK: Position mismatch detected between local DB and exchange.")
            return "BLOCK"

        # 6. Uncertainty & News Overrides
        # E.g., if TimesFM uncertainty penalty was extremely high, or news confidence is low during a major event.
        if is_major_event and news_confidence < 0.8:
            logger.warning("BLOCK: Low news confidence during a major event.")
            return "BLOCK"
            
        # 7. REDUCE check
        # If the market is moderately volatile but not breaching hard limits, we can REDUCE size.
        if market.spread_bps > (self.max_spread_bps * 0.7):
            logger.info("REDUCE: Market spread elevated, reducing position size.")
            self.processed_signals.add(signal_id)
            return "REDUCE"

        # If all checks pass
        self.processed_signals.add(signal_id)
        return "ALLOW"

    def activate_kill_switch(self) -> None:
        logger.critical("KILL SWITCH ACTIVATED.")
        self.kill_switch_active = True
