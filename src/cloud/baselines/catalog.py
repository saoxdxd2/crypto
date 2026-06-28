from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BaselineStrategy:
    name: str
    class_name: str
    description: str


BASELINE_STRATEGIES = [
    BaselineStrategy(
        name="buy_and_hold",
        class_name="BuyAndHoldBaseline",
        description="Enter once and hold until the backtest closes.",
    ),
    BaselineStrategy(
        name="ma_crossover",
        class_name="MaCrossoverBaseline",
        description="Long when fast moving average crosses above slow moving average.",
    ),
    BaselineStrategy(
        name="momentum",
        class_name="MomentumBaseline",
        description="Long when trailing return and trend filter are both positive.",
    ),
    BaselineStrategy(
        name="rsi_mean_reversion",
        class_name="RsiMeanReversionBaseline",
        description="Long after oversold RSI mean-reversion setup.",
    ),
    BaselineStrategy(
        name="random_entry",
        class_name="RandomEntryBaseline",
        description="Deterministic pseudo-random entry baseline for edge comparison.",
    ),
]


def baseline_catalog() -> list[dict[str, str]]:
    return [asdict(strategy) for strategy in BASELINE_STRATEGIES]
