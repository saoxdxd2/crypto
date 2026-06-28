import json
import logging
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DataWorker:
    """
    Background worker that polls local JSON files and trade history,
    then emits updates via callbacks to the GUI thread.
    """

    def __init__(
        self,
        signal_callback: Callable[[dict[str, Any]], None],
        news_callback: Callable[[dict[str, Any]], None],
        thresholds_callback: Callable[[dict[str, Any]], None],
        trades_callback: Callable[[list[dict[str, Any]]], None],
        thinking_callback: Callable[[list[dict[str, Any]]], None],
        poll_interval_sec: float = 1.0,
    ):
        self.signal_callback = signal_callback
        self.news_callback = news_callback
        self.thresholds_callback = thresholds_callback
        self.trades_callback = trades_callback
        self.thinking_callback = thinking_callback
        self.poll_interval_sec = poll_interval_sec
        self.stop_event = Event()

        self.data_dir = Path("data")
        self.signal_file = self.data_dir / "signals" / "LOBERT_1m.json"
        self.news_file = self.data_dir / "metadata" / "regime.json"
        self.thresholds_file = self.data_dir / "metadata" / "optimal_thresholds.json"
        self.trades_file = self.data_dir / "trades" / "trade_history.json"
        self.thinking_file = self.data_dir / "signals" / "thinking_log.json"

        self._thread = Thread(target=self._run_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._thread.join(timeout=2.0)

    def _safe_read_json(self, path: Path) -> dict[str, Any] | list:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            # logger.warning(f"Error reading {path.name}: {e}")
            return {}

    def _run_loop(self) -> None:
        from filelock import FileLock
        while not self.stop_event.is_set():
            # 1. Read latest LOBERT signal
            if self.signal_file.exists():
                try:
                    with FileLock(f"{self.signal_file}.lock", timeout=1):
                        signals = self._safe_read_json(self.signal_file)
                        if signals and isinstance(signals, list):
                            last_sig = signals[-1]
                            # Map to GUI format
                            gui_sig = {
                                "net_edge": last_sig.get("book_imbalance", 0.0),
                                "action": "OPEN" if "BULL" in last_sig.get("pattern", "") else ("CLOSE" if "BEAR" in last_sig.get("pattern", "") else "--"),
                                "size": "0.1",
                                "reason_code": last_sig.get("pattern", "--")
                            }
                            self.signal_callback(gui_sig)
                except Exception:
                    pass

            # 2. Read latest Regime/News Risk Modifier
            if self.news_file.exists():
                try:
                    with FileLock(f"{self.news_file}.lock", timeout=1):
                        news = self._safe_read_json(self.news_file)
                        if news and isinstance(news, dict):
                            gui_news = {
                                "risk_modifier": "1.0" if news.get("is_safe", True) else "0.0",
                                "summary": news.get("regime_summary", "No recent events.")
                            }
                            self.news_callback(gui_news)
                except Exception:
                    pass

            # 3. Read latest FLAML Thresholds
            thresholds = self._safe_read_json(self.thresholds_file)
            if thresholds and isinstance(thresholds, dict):
                self.thresholds_callback(thresholds)

            # 4. Read trade history
            trades = self._safe_read_json(self.trades_file)
            if isinstance(trades, list):
                self.trades_callback(trades)

            # 5. Read thinking / decision log
            thinking = self._safe_read_json(self.thinking_file)
            if isinstance(thinking, list):
                self.thinking_callback(thinking)

            # Wait for next poll
            time.sleep(self.poll_interval_sec)
