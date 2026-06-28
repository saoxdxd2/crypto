import sys
import os
import logging
from pathlib import Path

# Setup simple logging to console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from src.mission_control.deep_reasoning import DeepReasoningSpecialist

def test_loop_engineering():
    print("Initializing Deep Reasoning Specialist...")
    # Force model path relative to project root
    dr = DeepReasoningSpecialist(model_dir=Path("data/models"))
    
    print("\n--- SIMULATING CONTRADICTORY SCENARIO ---")
    print("Macro: Bearish (-0.05)")
    print("Micro: Bullish Sweep (0.8)")
    print("News: SEC approves new regulatory crackdown (Highly Bearish)")
    
    print("\n--- STARTING GENERATOR/EVALUATOR LOOP ---")
    
    # We pass contradictory inputs. 
    # If the Generator says it's "safe" just because Micro is bullish, 
    # the Evaluator should catch it because Macro and News are both Bearish.
    result = dr.analyze_market_regime(
        symbol="BTCUSDT",
        timesfm_edge=-0.0500,
        pattern_score=0.85,
        news_summary="SEC announces massive regulatory crackdown on all cryptocurrency exchanges, causing panic."
    )
    
    print("\n--- FINAL DECISION ---")
    print(result)

if __name__ == "__main__":
    test_loop_engineering()
