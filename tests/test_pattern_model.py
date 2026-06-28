"""
Performance verification for the PatchTST pattern model.
Checks CPU inference latency and fine-tuning speed.
"""
import time
import logging
from crypto_research.pattern_model import PatchTSTPatternModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pattern_model_performance():
    model = PatchTSTPatternModel()
    
    # 1. Test Inference Latency
    logger.info("Testing inference latency (target < 50ms)...")
    dummy_returns = [0.001] * 512
    
    # warmup
    for _ in range(5):
        model.predict_pattern_score(dummy_returns)
        
    times = []
    for _ in range(50):
        t0 = time.perf_counter_ns()
        score = model.predict_pattern_score(dummy_returns)
        times.append((time.perf_counter_ns() - t0) / 1_000_000) # ms
        
    median_latency = sorted(times)[25]
    logger.info(f"Median inference latency: {median_latency:.2f} ms")
    assert median_latency < 50.0, f"Inference too slow: {median_latency:.2f} ms"
    
    # 2. Test Fine-Tuning Speed
    logger.info("Testing fine-tune latency (target < 5s for batch of 32)...")
    windows = [[0.001] * 512 for _ in range(32)]
    labels = [1] * 16 + [0] * 16
    
    # warmup
    model.finetune_step(windows[:2], labels[:2])
    
    t0 = time.perf_counter()
    loss = model.finetune_step(windows, labels)
    duration = time.perf_counter() - t0
    
    logger.info(f"Fine-tune pass took: {duration:.3f} s (loss={loss:.4f})")
    assert duration < 5.0, f"Fine-tuning too slow: {duration:.3f} s"
    
    print("ALL TESTS PASSED: Model fits CPU budget.")

if __name__ == "__main__":
    test_pattern_model_performance()
