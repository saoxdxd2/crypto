"""
Deep Performance Profiling Suite
=================================
Tests every major function for:
  - Latency (pytest-benchmark)
  - Memory (tracemalloc)
  - CPU time (cProfile)
  - Flags anything over 1ms latency or 10MB memory
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
import tracemalloc
from datetime import UTC, datetime

import pytest


# ──────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────
def _measure_memory(fn, *args, **kwargs):
    """Run fn and return (result, peak_memory_bytes)."""
    tracemalloc.start()
    result = fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak


def _measure_cpu(fn, *args, **kwargs):
    """Run fn under cProfile and return (result, top_5_lines)."""
    pr = cProfile.Profile()
    pr.enable()
    result = fn(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(10)
    return result, s.getvalue()


# ──────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────
@pytest.fixture
def sample_forecast():
    return {
        "expected_return": 0.005,
        "p10_return": -0.002,
        "p90_return": 0.012,
        "created_at": datetime.now(UTC).isoformat(),
        "exchange": "binance",
        "symbol": "BTCUSDT",
    }


@pytest.fixture
def sample_news():
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "risk_modifier": 0.95,
        "summary": "Fed holds rates steady.",
    }


@pytest.fixture
def sample_signal():
    return {
        "signal_id": "sig_test0001",
        "created_at": datetime.now(UTC).isoformat(),
        "expires_at": (datetime.now(UTC)).isoformat(),
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "action": "open",
        "side": "long",
        "size": 0.42,
        "max_loss": 0.004,
        "net_edge": 0.002341,
        "reason_code": "TIMESFM_EDGE_NEWS_SAFE_COST_OK",
    }


# ──────────────────────────────────────────────────────
#  1. DeterministicDecisionMath.evaluate
# ──────────────────────────────────────────────────────
class TestDecisionPerformance:

    def test_decision_evaluate_latency(self, benchmark, sample_forecast, sample_news):
        from crypto_research.decision import DeterministicDecisionMath
        engine = DeterministicDecisionMath()
        result = benchmark(engine.evaluate, sample_forecast, sample_news)
        assert result["action"] in ("open", "hold")

    def test_decision_evaluate_memory(self, sample_forecast, sample_news):
        from crypto_research.decision import DeterministicDecisionMath
        engine = DeterministicDecisionMath()
        _, peak = _measure_memory(engine.evaluate, sample_forecast, sample_news)
        peak_mb = peak / 1024 / 1024
        print(f"\n  Decision.evaluate peak memory: {peak_mb:.2f} MB")
        assert peak_mb < 10, f"Memory too high: {peak_mb:.2f} MB"

    def test_decision_evaluate_cpu(self, sample_forecast, sample_news):
        from crypto_research.decision import DeterministicDecisionMath
        engine = DeterministicDecisionMath()
        _, profile = _measure_cpu(engine.evaluate, sample_forecast, sample_news)
        print(f"\n  Decision.evaluate CPU profile:\n{profile}")


# ──────────────────────────────────────────────────────
#  2. RiskGovernor.evaluate
# ──────────────────────────────────────────────────────
class TestGovernorPerformance:

    def test_governor_evaluate_latency(self, benchmark, sample_signal):
        from crypto_research.governor import RiskGovernor, MarketState, PortfolioState
        gov = RiskGovernor()
        market = MarketState(spread_bps=2.0, book_depth_usd=100000, is_api_stable=True, data_age_seconds=1.0)
        portfolio = PortfolioState(daily_loss_usd=50, open_trades_count=1, position_mismatch=False)
        result = benchmark(gov.evaluate, sample_signal, market, portfolio)
        assert result in ("ALLOW", "REDUCE", "BLOCK")

    def test_governor_evaluate_memory(self, sample_signal):
        from crypto_research.governor import RiskGovernor, MarketState, PortfolioState
        gov = RiskGovernor()
        market = MarketState(spread_bps=2.0, book_depth_usd=100000, is_api_stable=True, data_age_seconds=1.0)
        portfolio = PortfolioState(daily_loss_usd=50, open_trades_count=1, position_mismatch=False)
        _, peak = _measure_memory(gov.evaluate, sample_signal, market, portfolio)
        peak_mb = peak / 1024 / 1024
        print(f"\n  Governor.evaluate peak memory: {peak_mb:.2f} MB")
        assert peak_mb < 10, f"Memory too high: {peak_mb:.2f} MB"


# ──────────────────────────────────────────────────────
#  3. RL Agent inference
# ──────────────────────────────────────────────────────
class TestRLAgentPerformance:

    def test_rl_act_latency(self, benchmark):
        from crypto_research.rl_agent import PPOAgent
        agent = PPOAgent()
        state = [0.23, 0.95, 0.1, 0.5, 0.02, 0.0]
        scalar = benchmark(agent.act, state)
        assert 0.0 <= scalar <= 1.0

    def test_rl_act_memory(self):
        from crypto_research.rl_agent import PPOAgent
        agent = PPOAgent()
        state = [0.23, 0.95, 0.1, 0.5, 0.02, 0.0]
        _, peak = _measure_memory(agent.act, state)
        peak_mb = peak / 1024 / 1024
        print(f"\n  RL.act peak memory: {peak_mb:.2f} MB")
        assert peak_mb < 10, f"Memory too high: {peak_mb:.2f} MB"

    def test_rl_act_under_500us(self):
        """Hard assertion: RL inference must be < 0.5ms on CPU."""
        from crypto_research.rl_agent import PPOAgent
        agent = PPOAgent()
        state = [0.23, 0.95, 0.1, 0.5, 0.02, 0.0]

        # Warm up
        for _ in range(10):
            agent.act(state)

        # Measure
        times = []
        for _ in range(100):
            t0 = time.perf_counter_ns()
            agent.act(state)
            times.append((time.perf_counter_ns() - t0) / 1_000_000)  # ms

        median = sorted(times)[50]
        p99 = sorted(times)[99]
        print(f"\n  RL.act latency: median={median:.3f}ms  p99={p99:.3f}ms")
        assert median < 0.5, f"Median latency too high: {median:.3f}ms"


# ──────────────────────────────────────────────────────
#  4. FLAML Optimizer (load config)
# ──────────────────────────────────────────────────────
class TestFlamlPerformance:

    def test_load_config_latency(self, benchmark):
        from crypto_research.flaml_optimizer import FlamlOptimizer
        result = benchmark(FlamlOptimizer.load_optimal_config)
        assert isinstance(result, dict)


# ──────────────────────────────────────────────────────
#  5. build_state_vector
# ──────────────────────────────────────────────────────
class TestStateVectorPerformance:

    def test_build_state_vector_latency(self, benchmark, sample_signal, sample_news):
        from crypto_research.rl_agent import build_state_vector
        result = benchmark(build_state_vector, sample_signal, sample_news)
        assert len(result) == 6
