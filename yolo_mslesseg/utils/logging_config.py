"""
Script: logging_config.py

Description:
    Configures the global logging system used throughout the pipeline.
    Provides:
        - Custom logging levels:
              * SKIP   → to indicate that a result already exists (⏩).
              * HEADER → for clear stage headers (bold).
        - Colour-coded console output using ANSI escape codes.
        - Clean file logging ('pipeline.log') with ANSI codes stripped.
        - Unified get_logger() function for obtaining a per-script logger.

Usage:
    from yolo_mslesseg.utils.logging_config import get_logger
    logger = get_logger(__file__)

Conventions:
    - All configuration is performed once in this module.
    - All scripts must obtain their logger through get_logger().
    - The pipeline.log file is overwritten on each new pipeline execution.
"""

import logging
import re
import sys
from pathlib import Path

# ============================================================
#                   CUSTOM LEVELS
# ============================================================


def register_custom_level(value, name):
    """Register a custom logging level and add logger.<name_lowercase>()."""
    logging.addLevelName(value, name)

    def log_method(self, message, *args, **kwargs):
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
    """Formatter with ANSI colours for console output."""

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

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


class NoColorFormatter(logging.Formatter):
    """Formatter that strips ANSI codes before writing to a file."""

    def format(self, record):
        raw = super().format(record)
        return ANSI_ESCAPE.sub("", raw)


# ============================================================
#               GLOBAL LOGGING CONFIGURATION
# ============================================================


def configure_logging(level=logging.INFO, log_file=None):
    """
    Configure global logging:
        - Colour-coded handler for console output with UTF-8 encoding.
        - Plain handler for file output with UTF-8 encoding.
        - Custom SKIP and HEADER levels enabled.

    This function is called automatically when this module is imported.
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


# Configure logging when this module is imported
configure_logging()


def configure_demo_logging():
    """
    Configure logging for demo execution.
    """
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


def get_logger(source_file):
    """
    Return a script-specific logger.

    Args:
        source_file (str | Path): path of the script (__file__ recommended).

    Example:
        logger = get_logger(__file__)
    """
    name = Path(source_file).stem
    return logging.getLogger(name)
