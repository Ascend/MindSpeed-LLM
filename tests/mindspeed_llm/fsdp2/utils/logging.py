"""
Global logging configuration module with custom distributed logger (rank0 support).
"""
import logging
import sys
import os
import threading
from functools import lru_cache
from typing import Optional

# --------------------------
# Global Variables
# --------------------------
# Thread lock for thread-safe logging configuration (prevents duplicate handler creation)
_thread_lock = threading.Lock()
# Default logging handler instance (reused to avoid duplicate log output)
_default_handler = None
# Flag to mark whether the logging system has been initialized
_initialized = False


# --------------------------
# Custom Logger Implementation
# --------------------------
class _Logger(logging.Logger):
    """Custom logger class with rank0-specific logging methods and plain message output.

    This logger extends the standard logging.Logger to support:
    1. Rank0-specific logging (only log on the main process in distributed scenarios)
    2. One-time warning logging (avoid duplicate warning messages)
    3. Plain message logging (output without level/time prefix)
    """
    def info_rank0(self, msg: str, *args, **kwargs) -> None:
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            # Adjust stack level to point to user code (skip current method)
            kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
            self.info(msg, *args, **kwargs)

    def warn_rank0(self, msg: str, *args, **kwargs) -> None:
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
            self.warning(msg, *args, **kwargs)

    @lru_cache(None)
    def warn_once(self, msg: str, *args, **kwargs) -> None:
        """Log warning level message only once (lifetime) and only on the process with LOCAL_RANK=0.
        """
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
            self.warning_rank0(msg, *args, **kwargs)

    def debug_rank0(self, msg: str, *args, **kwargs) -> None:
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
            self.debug(msg, *args, **kwargs)

    def log_plain(self, msg: str, *args, **kwargs) -> None:
        """Core method for plain message logging without any prefix.

        Writes directly to stdout to avoid the standard logging format.

        Args:
            msg (str): Raw message to be logged (supports string formatting with args).
        """
        # Format message with arguments to keep consistency with standard logger
        formatted_msg = msg % args if args else msg
        # Write to stdout and flush immediately to ensure message output
        sys.stdout.write(f"{formatted_msg}\n")
        sys.stdout.flush()

    def info_plain(self, msg: str, *args, **kwargs) -> None:
        self.log_plain(msg, *args, **kwargs)

    def info_plain_rank0(self, msg: str, *args, **kwargs) -> None:
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            self.log_plain(msg, *args, **kwargs)


# Replace the default logger class with custom _Logger for global usage
logging.setLoggerClass(_Logger)


# --------------------------
# Internal Helper Functions
# --------------------------
def _get_library_name() -> str:
    """Get the root name of the library (first part of the module name).

    Returns:
        str: Root name of the current library module.
    """
    return __name__.split(".")[0]


def _get_library_root_logger() -> _Logger:
    """Get the root logger instance of the library (custom _Logger type).

    Returns:
        _Logger: Custom logger instance for the library root.
    """
    return logging.getLogger(_get_library_name())


# --------------------------
# Global Logging Configuration
# --------------------------
def setup_global_logging(level: str = "INFO") -> None:
    """Configure the global logging system with rank0 support and dynamic format.

    This function initializes the logging system with thread-safe configuration,
    dynamic log format (DEBUG level includes path/line number), and avoids duplicate handlers.
    It should be called only once at the project startup (main entry file).

    Args:
        level (str, optional): Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL). Defaults to "INFO".
    """
    global _default_handler, _initialized

    # Return immediately if logging system is already initialized
    if _initialized:
        return

    # Validate and convert the input log level to logging module constants
    valid_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = valid_levels.get(level.strip().upper(), logging.INFO)

    # Set dynamic log format based on log level
    if log_level == logging.DEBUG:
        log_format = "%(levelname)s [%(asctime)s %(pathname)s:%(lineno)s] >> %(message)s"
    else:
        log_format = "%(levelname)s [%(asctime)s] >> %(message)s"

    # Thread-safe configuration to avoid duplicate handlers
    with _thread_lock:
        # Configure global basic logging settings
        logging.root.handlers.clear()
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
            force=True # Override any existing basicConfig (Python 3.8+)
        )

        # Configure library-specific logger for backward compatibility
        library_root_logger = _get_library_root_logger()
        library_root_logger.handlers.clear()
        
        # Create or reuse the default handler with consistent format
        if not _default_handler:
            formatter = logging.Formatter(
                fmt=log_format,
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            _default_handler = logging.StreamHandler(sys.stdout)
            _default_handler.setFormatter(formatter)
        
        # Add handler to library logger and set propagation to avoid duplicate logs
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(log_level)
        library_root_logger.propagate = False
        # Mark logging system as initialized to prevent re-initialization
        _initialized = True

    # Log initialization completion message only on LOCAL_RANK=0
    library_root_logger.info_rank0(
        f"Global logging initialized | Level: {level.upper()} | Local Rank: {os.getenv('LOCAL_RANK', '0')}"
    )


# --------------------------
# Public Helper Functions
# --------------------------
def get_logger(name: Optional[str] = None, level: Optional[str] = "INFO") -> _Logger:
    """Get a custom _Logger instance with rank0 support.

    Ensures the global logging configuration is applied before returning the logger instance.
    Uses INFO as the default log level if not specified, or overrides with the given level.

    Args:
        name (Optional[str], optional): Name of the logger, defaults to library root name if None.
        level (Optional[str], optional): Log level for the logger (DEBUG/INFO/WARNING/ERROR/CRITICAL). 
            Defaults to "INFO".

    Returns:
        _Logger: Custom logger instance with rank0 logging methods.
    """
    if name is None:
        name = _get_library_name()
    
    # Apply global configuration
    setup_global_logging(level=level)
    return logging.getLogger(name)