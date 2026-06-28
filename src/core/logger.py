import logging
import sys
from pathlib import Path
from rich.logging import RichHandler

def setup_logger(name: str = "crypto_engine", log_file: str = "system.log") -> logging.Logger:
    """
    Creates a centralized, structured enterprise logger.
    Outputs to both the Rich Console (stdout) and a rolling log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger

    # Force UTF-8 on Windows terminal for emojis
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    # Ensure log directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 1. Console Handler (Rich formatting)
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
    )
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_fmt)
    
    # 2. File Handler (Structured text with Rotation)
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_dir / log_file, 
        maxBytes=10 * 1024 * 1024,  # 10 MB limit
        backupCount=3,              # Keep 3 old logs
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "msg": "%(message)s"}'
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Global exception hook to log unhandled crashes
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught Exception!", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    return logger

# Singleton default logger
logger = setup_logger()
