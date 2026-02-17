import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_file = _LOG_DIR / f"{name.replace('.', '_')}_{datetime.now():%Y%m%d}.log"
    fh = logging.handlers.RotatingFileHandler(
        str(log_file), maxBytes=10 * 1024 * 1024, backupCount=5
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
