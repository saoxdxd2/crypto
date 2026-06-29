"""
Deep Reasoning Specialist (GGUF LLM)

Uses a 500M parameter Language Model quantized to 4-bit (GGUF format)
running via llama.cpp. This model is used strictly for INFERENCE
(overseeing the strategy and determining market regime) and is NOT
fine-tuned online, allowing it to run fast on CPU without stalling the pipeline.

Model used: Qwen2.5-0.5B-Instruct-Q4_K_M
"""
import logging
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from src.core.logger import logger

# The Arbiter: Highly capable 0.6B parameter instruction-tuned model in 4-bit
REPO_ID = "unsloth/Qwen3-0.6B-GGUF"
FILENAME = "Qwen3-0.6B-Q4_K_M.gguf"


from src.core.sys_events import push_sys_event

class DeepReasoningSpecialist:
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or Path("data/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / FILENAME
        
        # We no longer load the model directly into RAM here.
        # Instead, we communicate with llm_server.py
        
    def _call_llm_server(self, prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
        """Helper to call the shared LLM server API."""
        import urllib.request
        try:
            data = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}).encode('utf-8')
            req = urllib.request.Request("http://localhost:5001/generate", data=data, headers={'Content-Type': 'application/json'}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("response", "")
        except Exception as e:
            logger.error(f"Failed to call LLM server: {e}")
            raise

    def _evaluate_decision(self, context: str, generator_output: str) -> tuple[bool, str]:
        """
        The Evaluator: Reviews the Generator's decision for logical consistency.
        Returns (is_approved, critique_if_rejected).
        """
        evaluator_prompt = f"""<|im_start|>system
You are the Verifier for an automated trading desk. Your job is to review the Tactical Arbiter's proposed decision for logical safety.
<|im_end|>
<|im_start|>user
Original Context:
{context}

Arbiter's Proposed Output:
{generator_output}

Task: Determine if the Arbiter's logic holds up. If the Macro and Micro indicators conflict severely, the Arbiter MUST NOT output "is_safe": true unless the News context provides an overwhelmingly bullish catalyst. 
If the logic is flawed, hallucinated, or overly risky, output REJECT and explain why. If it is sound, output APPROVE.

Output exactly in this JSON format:
{{
  "verdict": "APPROVE or REJECT",
  "critique": "If REJECT, explain why the logic is flawed. If APPROVE, leave empty."
}}
<|im_end|>
<|im_start|>assistant
"""
        try:
            text = self._call_llm_server(evaluator_prompt, max_tokens=150, temperature=0.1)
            
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(text)
            verdict = result.get("verdict", "REJECT")
            critique = result.get("critique", "Verification parsing error.")
            
            return (verdict == "APPROVE", critique)
        except Exception as e:
            logger.error(f"Evaluator LLM failed to parse response: {e}")
            return (False, f"Evaluator crashed: {str(e)}")

    def analyze_market_regime(
        self,
        symbol: str,
        timesfm_edge: float,
        pattern_score: float,
        news_summary: str,
        portfolio_balance: float = 10000.0,
        user_settings: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Ask the LLM to analyze the current market state using a Generator/Evaluator loop.
        Returns a dict with 'regime_summary' and 'is_safe'.
        """
        if user_settings is None:
            user_settings = {}
            
        settings_str = ", ".join(f"{k}={v}" for k,v in user_settings.items()) if user_settings else "Default AI Limits"
        
        context = f"""Intelligence Brief for {symbol}:
1. The Macro Navigator (FinCast - 4H Regime): Expected Edge = {timesfm_edge:.4f} (>0 means macro uptrend)
2. The Tactical Sniper (LOBERT - 1m Order Book): Microstructure Score = {pattern_score:.2f} (>0.5 means stop-loss hunting or bullish sweep detected)
3. Latest News Context: {news_summary}
4. Current Portfolio Balance: ${portfolio_balance:,.2f}
5. Active Risk Constraints: {settings_str}"""

        base_prompt = f"""<|im_start|>system
You are the Tactical Arbiter for an automated trading desk. You ingest intelligence from specialized models and output ONLY valid JSON.
<|im_end|>
<|im_start|>user
{context}

Synthesize these inputs. If the Navigator and Sniper disagree, you must arbitrate based on the News context.
Factor in the Portfolio Balance and Risk Constraints. If the trade is excessively risky relative to the balance and constraints, DO NOT execute.
Is it safe to execute a trade right now, or are we in a liquidity trap/choppy regime?

Output exactly in this JSON format:
{{
  "regime_summary": "1 sentence synthesis of the Macro vs Micro conflict.",
  "is_safe": true or false
}}
<|im_end|>
"""
        
        feedback_history = ""
        max_retries = 3
        
        for attempt in range(max_retries):
            # 1. GENERATOR PASS
            prompt = base_prompt + feedback_history + "<|im_start|>assistant\n"
            
            try:
                text = self._call_llm_server(prompt, max_tokens=150, temperature=0.3)
                
                if text.startswith("```json"):
                    text = text.replace("```json", "").replace("```", "").strip()
                    
                generator_json = json.loads(text)
                
                # 2. EVALUATOR PASS (Loop Engineering Verification)
                logger.info(f"Loop Engineering: Evaluator analyzing attempt {attempt+1}/{max_retries}...")
                is_approved, critique = self._evaluate_decision(context, text)
                
                if is_approved:
                    logger.info(f"Loop Engineering: Evaluator APPROVED.")
                    push_sys_event("ALLOW", f"Loop Engineering Consensus reached: {generator_json.get('regime_summary', 'Safe')}")
                    
                    is_safe_val = bool(generator_json.get("is_safe", True))
                    # Write to decoupled state file for fast loops to read
                    regime_path = Path("data/metadata/regime.json")
                    regime_path.parent.mkdir(parents=True, exist_ok=True)
                    regime_path.write_text(json.dumps({"is_safe": is_safe_val, "regime_summary": generator_json.get("regime_summary", "")}))
                    
                    return {
                        "regime_summary": generator_json.get("regime_summary", "Unknown regime"),
                        "is_safe": is_safe_val
                    }
                else:
                    logger.warning(f"Loop Engineering: Evaluator REJECTED. Critique: {critique}")
                    push_sys_event("REJECT", f"Loop Engineering (Attempt {attempt+1}): Evaluator rejected Generator logic. Critique: {critique}")
                    # 3. FEEDBACK
                    feedback_history += f"<|im_start|>assistant\n{text}\n<|im_end|>\n<|im_start|>user\nThe Verifier REJECTED your logic with this critique: {critique}. Please revise your analysis and output a corrected JSON response.\n<|im_end|>\n"
                    
            except Exception as e:
                logger.error(f"DeepReasoning LLM failed during Generator pass: {e}")
                push_sys_event("ERROR", f"LLM parsing failed on attempt {attempt+1}: {str(e)}")
                feedback_history += f"<|im_start|>user\nYour previous JSON output was invalid. Error: {e}. Output exactly valid JSON this time.\n<|im_end|>\n"

        # 4. EXHAUSTED RETRIES FALLBACK
        logger.critical(f"Loop Engineering: Failed to reach consensus after {max_retries} attempts. Defaulting to BLOCK.")
        push_sys_event("BLOCK", "Loop Engineering exhausted retries without consensus. Fallback triggered.")
        
        regime_path = Path("data/metadata/regime.json")
        regime_path.parent.mkdir(parents=True, exist_ok=True)
        regime_path.write_text(json.dumps({"is_safe": False, "regime_summary": "Loop consensus failed. Safety triggered."}))
        
        return {
            "regime_summary": "Loop consensus failed. Safety triggered.",
            "is_safe": False # Fail-safe block
        }
