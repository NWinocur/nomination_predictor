"""Project configuration and directory paths.

This module defines the project's directory structure and ensures all required
directories exist when the module is imported.

Directory Structure:
- data/external: Raw data from third-party sources (HTML/PDF downloads)
- data/raw: Initial parsed data (CSVs from scraped HTML/PDFs)
- data/interim: Cleaned and transformed data
- data/processed: Final datasets ready for modeling
"""

import os
from pathlib import Path
import sys
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
) -> None:
    """Configure loguru logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, logs only to stderr.
        rotation: Log rotation condition (e.g., "10 MB", "1 day")
        retention: Log retention period (e.g., "30 days")
        format: Log message format
    """
    # Remove default handler
    logger.remove()
    
    # Add stderr handler with color support
    logger.add(
        sys.stderr,
        level=level,
        format=format,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler if log_file is specified
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_file),
            rotation=rotation,
            retention=retention,
            level=level,
            format=format,
            backtrace=True,
            diagnose=True,
            enqueue=True  # Makes logging thread-safe
        )
    
    # Configure tqdm integration if available
    try:
        from tqdm import tqdm
        
        def tqdm_write(msg):
            tqdm.write(msg, end="")
            
        logger.remove()  # Remove all handlers
        logger.add(tqdm_write, level=level, format=format, colorize=True)
        
        if log_file:
            logger.add(
                str(log_file),
                rotation=rotation,
                retention=retention,
                level=level,
                format=format,
                enqueue=True
            )
            
    except ImportError:
        pass  # tqdm not available, use standard logging

# Base paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Configure default logging
LOG_DIR = PROJ_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"
configure_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=LOG_FILE
)

logger.info(f"Project root: {PROJ_ROOT}")

# Data directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"        # Initial parsed data from scraped HTML/PDFs
INTERIM_DATA_DIR = DATA_DIR / "interim"  # Cleaned and transformed data
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Final datasets for modeling
EXTERNAL_DATA_DIR = DATA_DIR / "external"    # Raw downloaded files (HTML/PDFs)

# Other directories
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# fuzzy-matcher parameters; weights must add up to 1 or they may be ignored
NAME_WEIGHT = 0.97
COURT_WEIGHT = 0.01
DATE_WEIGHT = 0.02
MATCH_THRESHOLD = 0.7
AMBIGUITY_THRESHOLD = 0.9

# Ensure all directories exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, LOG_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")

# Log configuration complete
logger.info("Configuration loaded")
