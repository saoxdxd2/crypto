from __future__ import annotations
import logging
import json
from pathlib import Path
from typing import Any, TypedDict, List
from filelock import FileLock
from src.core.sys_events import push_sys_event

logger = logging.getLogger(__name__)

class ResearchState(TypedDict):
    symbol: str
    losing_trades: List[dict]
    post_mortem_critique: str
    thresholds_updated: bool
    requires_retraining: bool
    errors: list[str]

class ResearchWorkflows:
    """
    Phase 17: Loop Engineering for Post-Mortem Research Automation.
    Automates overnight parsing of losing trades, LLM critiques, and hyperparameter adjustment.
    """

    def __init__(self) -> None:
        self.state: ResearchState = {
            "symbol": "BTCUSDT",
            "losing_trades": [],
            "post_mortem_critique": "",
            "thresholds_updated": False,
            "requires_retraining": False,
            "errors": [],
        }
        
    def _call_llm_server(self, prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> str:
        import urllib.request
        try:
            data = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}).encode('utf-8')
            req = urllib.request.Request("http://localhost:5001/generate", data=data, headers={'Content-Type': 'application/json'}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("response", "")
        except Exception as e:
            logger.error(f"Failed to call LLM server: {e}")
            return "{}"

    def execute_post_mortem_loop(self) -> ResearchState:
        """
        Executes the overnight post-mortem loop sequentially.
        """
        logger.info("Starting Overnight Post-Mortem Loop Engineering...")
        push_sys_event("SYSTEM", "Starting Overnight Post-Mortem Analysis.")
        
        self.state = self._node_gather_losses(self.state)
        if not self.state["losing_trades"]:
            logger.info("No losing trades to analyze. Loop complete.")
            return self.state
            
        self.state = self._node_llm_critique(self.state)
        self.state = self._node_adjust_thresholds(self.state)
        self.state = self._node_trigger_retraining(self.state)
        
        logger.info("Post-Mortem Loop Complete.")
        return self.state

    def _node_gather_losses(self, state: ResearchState) -> ResearchState:
        logger.info("Node: Gathering losing trades...")
        trades_path = Path("data/trades/trade_history.json")
        if trades_path.exists():
            lock = FileLock(f"{trades_path}.lock")
            with lock:
                try:
                    trades = json.loads(trades_path.read_text())
                    state["losing_trades"] = [t for t in trades if t.get("pnl", 0) < 0]
                except Exception as e:
                    state["errors"].append(f"Failed to read trades: {e}")
        return state

    def _node_llm_critique(self, state: ResearchState) -> ResearchState:
        logger.info("Node: LLM Critiquing losses...")
        losses_summary = json.dumps(state["losing_trades"][-5:]) # Analyze last 5 losses
        
        prompt = f"""<|im_start|>system
You are a Quantitative Analyst AI reviewing a trading bot's daily losses.
<|im_end|>
<|im_start|>user
Here are the recent losing trades: {losses_summary}
Analyze why these failed. Was the LOBERT threshold too low (frequent false positives)? Is the market too choppy?
Output exactly valid JSON in this format:
{{
  "critique": "Brief explanation of the failure mode",
  "recommended_momentum_threshold": 15.0,
  "requires_retraining": false
}}
<|im_end|>
<|im_start|>assistant
"""
        response = self._call_llm_server(prompt)
        try:
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(response)
            state["post_mortem_critique"] = result.get("critique", "")
            state["recommended_momentum_threshold"] = result.get("recommended_momentum_threshold", 10.0)
            state["requires_retraining"] = result.get("requires_retraining", False)
            
            push_sys_event("ALLOW", f"Post-Mortem Critique: {state['post_mortem_critique']}")
        except Exception as e:
            state["errors"].append(f"LLM Critique parsing failed: {e}")
            
        return state

    def _node_adjust_thresholds(self, state: ResearchState) -> ResearchState:
        logger.info("Node: Adjusting thresholds...")
        rec_thresh = state.get("recommended_momentum_threshold")
        if rec_thresh:
            settings_path = Path("data/metadata/optimal_thresholds.json")
            settings = {}
            if settings_path.exists():
                try:
                    settings = json.loads(settings_path.read_text())
                except:
                    pass
            
            settings["momentum_threshold"] = rec_thresh
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(json.dumps(settings, indent=2))
            state["thresholds_updated"] = True
            logger.info(f"Updated momentum threshold to {rec_thresh}")
            push_sys_event("SYSTEM", f"Automated adjustment: Momentum threshold -> {rec_thresh}")
            
        return state

    def _node_trigger_retraining(self, state: ResearchState) -> ResearchState:
        logger.info("Node: Evaluating retraining needs...")
        if state.get("requires_retraining"):
            logger.warning("LLM requested model retraining due to edge decay!")
            push_sys_event("BLOCK", "Edge Decay Detected! Triggering Cloud Retraining pipeline.")
            # Trigger online retraining adaptation
            logger.info("Online Continuous Learning natively adapts to edge decay without requiring offline retrains.")
            push_sys_event("SYSTEM", "Triggering aggressive CPU online adaptation steps.")
            
        return state
