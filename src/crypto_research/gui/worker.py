import json
import logging
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable

logger = logging.getLogger(__name__)

class DataWorker:
    """
    Background worker that polls the local JSON files and emits updates via callbacks
    to ensure the GUI thread never blocks.
    """
    def __init__(
        self, 
        signal_callback: Callable[[dict[str, Any]], None],
        news_callback: Callable[[dict[str, Any]], None],
        thresholds_callback: Callable[[dict[str, Any]], None],
        poll_interval_sec: float = 1.0
    ):
        self.signal_callback = signal_callback
        self.news_callback = news_callback
        self.thresholds_callback = thresholds_callback
        self.poll_interval_sec = poll_interval_sec
        self.stop_event = Event()
        
        self.data_dir = Path("data")
        self.signal_file = self.data_dir / "signals" / "latest_signal.json"
        self.news_file = self.data_dir / "signals" / "latest_news_BTCUSDT.json"
        self.thresholds_file = self.data_dir / "metadata" / "optimal_thresholds.json"
        
        self._thread = Thread(target=self._run_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._thread.join(timeout=2.0)

    def _safe_read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Error reading {path.name}: {e}")
            return {}

    def _run_loop(self) -> None:
        while not self.stop_event.is_set():
            # 1. Read latest TimesFM signal
            signal = self._safe_read_json(self.signal_file)
            if signal:
                self.signal_callback(signal)

            # 2. Read latest News Risk Modifier
            news = self._safe_read_json(self.news_file)
            if news:
                self.news_callback(news)

            # 3. Read latest FLAML Thresholds
            thresholds = self._safe_read_json(self.thresholds_file)
            if thresholds:
                self.thresholds_callback(thresholds)

            # Wait for next poll
            time.sleep(self.poll_interval_sec)
