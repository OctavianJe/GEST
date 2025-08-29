import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from gest.common.helpers.config_loader import ConfigLoader, ConfigurationError


class LoggingHelper:
    """Singleton helper class for logging."""

    _instance: "Optional[LoggingHelper]" = None
    _lock = threading.Lock()

    def __new__(cls) -> "LoggingHelper":
        """Singleton pattern to ensure only one instance of LoggingHelper exists."""

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        """Initialize the logging configuration if not already initialized."""

        if getattr(self, "_initialized", False):
            return

        # Define the log format
        fmt = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
        formatter = logging.Formatter(fmt)

        # Get the log directory from the configuration
        log_dir = self._get_log_directory()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "gest.log"

        # Create a rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            mode="a",
            maxBytes=1 * 1024 * 1024,  # 1 MB
            backupCount=1000,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(formatter)

        # Create a stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Set the root logger to the default level from the configuration
        logging.basicConfig(
            level=self._get_default_level(),
            handlers=[stream_handler, file_handler],
            force=False,
        )

        self._initialized = True

    def _get_default_level(self) -> int:
        """Get the default logging level from the configuration."""

        level = ConfigLoader().get("gest.logging.level")

        if not isinstance(level, str):
            raise ConfigurationError(
                "Invalid logging level in configuration. Expected a string."
            )

        if level not in logging._nameToLevel:
            raise ConfigurationError(
                f"Invalid '{level}' logging level. Available levels are {list(logging._nameToLevel.keys())}"
            )

        return logging._nameToLevel[level]

    def _get_log_directory(self) -> Path:
        """Get the default log directory from the configuration."""

        log_dir = ConfigLoader().get("gest.logging.dir")

        if not isinstance(log_dir, str):
            raise ConfigurationError(
                "Invalid directory in configuration. Expected a string path."
            )

        return Path(log_dir)
