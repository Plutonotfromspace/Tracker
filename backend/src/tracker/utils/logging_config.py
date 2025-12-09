"""
Centralized logging configuration for the vehicle tracking system.

Provides consistent logging across all modules with UTF-8 encoding support
for Windows compatibility.
"""

import logging
import sys
import io
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = "outputs/logs/tracker.log",
    level: int = logging.INFO,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the vehicle tracking system.
    
    Args:
        log_file: Path to log file. If None, only console output is used.
        level: Logging level (default: logging.INFO).
        log_format: Custom format string. If None, uses default format.
    
    Returns:
        Configured root logger.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers: list[logging.Handler] = []
    
    # File handler with UTF-8 encoding
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # Silence noisy third-party loggers (SAM2, transformers, etc.)
    noisy_loggers = [
        "sam2",
        "segment_anything",
        "transformers",
        "PIL",
        "urllib3",
        "httpx",
        "httpcore",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Also silence the root logger's SAM2 spam (comes through root)
    # SAM2 logs directly to root logger, so we need to filter those
    
    # Fix console encoding for Windows
    _fix_windows_encoding()
    
    return logging.getLogger()


def _fix_windows_encoding() -> None:
    """Fix console encoding for Windows to handle special characters."""
    if sys.platform == "win32":
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        elif hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, 
                encoding='utf-8', 
                errors='replace'
            )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__).
    
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


# Module-level logger
logger = get_logger(__name__)
