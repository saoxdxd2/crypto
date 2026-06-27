import pytest
import numpy as np
from datetime import datetime, UTC, timedelta
from crypto_research.decision import DeterministicDecisionMath

@pytest.fixture
def sample_forecast():
    return {
        "forecast_id": "test",
        "created_at": datetime.now(UTC).isoformat(),
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "expected_return": 0.005,
        "p10_return": -0.002,
        "p90_return": 0.012
    }

@pytest.fixture
def sample_news():
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "risk_modifier": 0.8
    }

def test_decision_math_performance(benchmark, sample_forecast, sample_news):
    """
    Benchmark the latency of the deterministic decision module in Python.
    If this takes > 1ms, it might need porting to C++/Zig.
    """
    decision_module = DeterministicDecisionMath()
    
    def run_eval():
        return decision_module.evaluate(sample_forecast, sample_news)
        
    result = benchmark(run_eval)
    assert result["action"] in ["open", "hold", "close"]

def test_array_creation_overhead(benchmark):
    """
    Benchmark standard numpy array overhead.
    """
    def run_numpy():
        return np.random.randn(10000) * 2.0
        
    benchmark(run_numpy)
