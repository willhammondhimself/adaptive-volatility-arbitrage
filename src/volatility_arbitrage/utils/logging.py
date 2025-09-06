"""
Structured JSON logging for the backtesting engine.

Provides consistent, machine-readable logging across all modules.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from volatility_arbitrage.core.config import LoggingConfig


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Formats log records as JSON objects with consistent fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter.

    Provides colorized output for console logging.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as colored text.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        level = f"{color}{record.levelname:8s}{reset}"
        logger = f"{record.name:20s}"
        message = record.getMessage()

        log_line = f"{timestamp} | {level} | {logger} | {message}"

        # Add exception info if present
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"

        return log_line


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Configure logging based on configuration.

    Args:
        config: Logging configuration. If None, uses defaults.
    """
    if config is None:
        from volatility_arbitrage.core.config import LoggingConfig

        config = LoggingConfig()

    # Create log directory if it doesn't exist
    config.log_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    if config.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level))

        if config.format == "json":
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(TextFormatter())

        root_logger.addHandler(console_handler)

    # File handler - always JSON for files
    log_file = config.log_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, config.level))
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter for adding contextual information.

    Allows adding extra fields to all log records from a logger.
    """

    def process(
        self, msg: str, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Process log message and add extra fields.

        Args:
            msg: Log message
            kwargs: Keyword arguments for logging

        Returns:
            Processed message and kwargs
        """
        # Merge extra fields
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra

        return msg, kwargs


def get_contextual_logger(
    name: str, **context: Any
) -> LoggerAdapter:
    """
    Get a logger with contextual information.

    Args:
        name: Logger name
        **context: Contextual key-value pairs to include in all logs

    Returns:
        Logger adapter with context

    Example:
        >>> logger = get_contextual_logger(__name__, strategy="vol_arb", symbol="SPY")
        >>> logger.info("Trade executed")
        # Logs will include strategy and symbol fields
    """
    base_logger = get_logger(name)
    return LoggerAdapter(base_logger, {"extra_fields": context})
