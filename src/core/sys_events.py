import json
import logging
from pathlib import Path
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

def push_sys_event(verdict: str, reason: str, progress: float = None):
    """
    Pushes an event directly to the thinking_log.json so the GUI can display it
    in the 'Decision Reasoning' panel. Useful for async tasks like model downloads.
    """
    path = Path("data/signals/thinking_log.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    entries = []
    if path.exists():
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Error reading thinking_log.json: {e}")
            
    now = datetime.now(UTC).strftime("%H:%M:%S")
    entry = {"time": now, "verdict": verdict, "reason": reason}
    if progress is not None:
        entry["progress"] = progress
    entries.append(entry)
    
    # keep last 20
    entries = entries[-20:]
    try:
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to write to thinking_log.json: {e}")
