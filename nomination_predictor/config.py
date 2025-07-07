"""Project configuration and directory paths.

This module defines the project's directory structure and ensures all required
directories exist when the module is imported.

Directory Structure:
- data/external: Raw data from third-party sources (HTML/PDF downloads)
- data/raw: Initial parsed data (CSVs from scraped HTML/PDFs)
- data/interim: Cleaned and transformed data
- data/processed: Final datasets ready for modeling
"""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Base paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

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

# Ensure all directories exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
