"""
Module: logging_config.py

Description:
    Configures the global logging system used throughout the pipeline.
    Provides:
        - Custom logging levels:
              * SKIP   → to indicate that a result already exists (⏩).
              * HEADER → for clear stage headers (bold).
        - Colour-coded console output using ANSI escape codes.
        - Clean file logging ('pipeline.log') with ANSI codes stripped.
        - Unified get_logger() function for obtaining a per-script logger.

    All configuration is performed once on module import. All scripts
    must obtain their logger through get_logger(). The pipeline.log file
    is overwritten on each new pipeline execution.

Usage:
    from yolo_mslesseg.utils.logging_config import get_logger
    logger = get_logger(__file__)

Inputs:
    None. Configuration is performed at import time.

Outputs:
    None. Registers handlers on the root logger.

Relationships:
    - Imported by all pipeline scripts and utility modules.
    - Used by ConfigBase and its subclasses.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any

# ============================================================
#                   CUSTOM LEVELS
# ============================================================


def register_custom_level(value: int, name: str) -> int:
    """Registers a custom logging level and adds a logger.<name_lowercase>() method.

    Args:
        value: Numeric level value (must be unique among registered levels).
        name: Level name string (stored in uppercase by the logging module).

    Returns:
        The level value passed in, for use as a module-level constant.
    """
    logging.addLevelName(value, name)

    def log_method(self, message: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(value):
            self._log(value, message, args, **kwargs)

    setattr(logging.Logger, name.lower(), log_method)
    return value


# Additional levels
SKIP_LEVEL = register_custom_level(23, "SKIP")    # ⏩ results already exist
HEADER_LEVEL = register_custom_level(35, "HEADER")  # Stage headers

# Regular expression for stripping ANSI codes
ANSI_ESCAPE = re.compile(r"\x1B\[[0-?][ -/][@-~]")

# ============================================================
#                     FORMATTERS
# ============================================================


class ColorFormatter(logging.Formatter):
    """
    Class: ColorFormatter

    Description:
        logging.Formatter subclass that wraps each log message in ANSI
        colour codes based on the severity level. Used for the console
        (stdout) handler to improve readability during pipeline execution.

    Attributes:
        COLORS (dict[int, str]):
            Mapping from logging level integers to ANSI colour escape codes.
        RESET (str):
            ANSI reset sequence appended after every formatted message.
    """

    COLORS = {
        logging.DEBUG: "\033[90m",        # Grey
        logging.INFO: "\033[38;5;39m",    # Bright blue
        logging.WARNING: "\033[1;93m",    # Bold yellow
        logging.ERROR: "\033[1;91m",      # Bold red
        logging.CRITICAL: "\033[1;97;41m",  # Bold white on red background
        SKIP_LEVEL: "\033[38;5;33m",      # Blue
        HEADER_LEVEL: "\033[1;97m",       # Bold white
    }

    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record by wrapping the message in the appropriate ANSI colour codes.

        Args:
            record: Log record to format.

        Returns:
            Formatted log message string with ANSI colour prefix and reset suffix.
        """
        color = self.COLORS.get(record.levelno, self.RESET)
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


class NoColorFormatter(logging.Formatter):
    """
    Class: NoColorFormatter

    Description:
        logging.Formatter subclass that removes ANSI escape sequences from
        log messages before writing them to a file. Used for the file handler
        (pipeline.log) to produce clean plain-text log output.

    Attributes:
        None beyond those inherited from logging.Formatter.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record by stripping all ANSI escape sequences from the message.

        Args:
            record: Log record to format.

        Returns:
            Formatted log message string with ANSI codes removed.
        """
        raw = super().format(record)
        return ANSI_ESCAPE.sub("", raw)


# ============================================================
#               GLOBAL LOGGING CONFIGURATION
# ============================================================


def configure_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Configures global logging with a colour-coded console handler and optional file handler.

    Sets up the root logger with ANSI-colour output to stdout and, when a path
    is supplied, a plain-text file handler. Custom SKIP and HEADER levels are
    automatically available after this call. Called automatically on module import.

    Args:
        level: Minimum logging level for the root logger. Defaults to logging.INFO.
        log_file: Optional path for a plain-text log file. No file handler is
            added when None.

    Returns:
        Configured root logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    # 1. Console handler with colours and UTF-8
    # Reconfigure sys.stdout to use UTF-8 on Windows
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            import codecs

            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColorFormatter("%(message)s"))
    logger.addHandler(ch)

    # 2. File handler without colours with UTF-8
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(NoColorFormatter("%(message)s"))
        logger.addHandler(fh)

    return logger


configure_logging()


def configure_demo_logging() -> None:
    """Configures logging for demo execution by replacing the pipeline.log handler with demo.log."""
    logger = logging.getLogger()

    # Remove only the FileHandler pointing to pipeline.log
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            # Avoid accidentally removing other future handlers
            if "pipeline.log" in str(getattr(h, "baseFilename", "")):
                logger.removeHandler(h)

    # demo.log relative to the current working directory (demo/)
    demo_log_path = Path.cwd() / "demo.log"

    # Add FileHandler for demo.log with UTF-8
    demo_handler = logging.FileHandler(demo_log_path, mode="w", encoding="utf-8")
    demo_handler.setLevel(logging.INFO)
    demo_handler.setFormatter(NoColorFormatter("%(message)s"))
    logger.addHandler(demo_handler)


# ============================================================
#               PUBLIC FUNCTION FOR SCRIPTS
# ============================================================


def get_logger(source_file: str | Path) -> logging.Logger:
    """Returns a script-specific logger named after the source file stem.

    Args:
        source_file: Path of the calling script. Pass __file__ to derive the
            logger name from the module filename.

    Returns:
        Logger instance named after the stem of source_file.
    """
    name = Path(source_file).stem
    return logging.getLogger(name)
