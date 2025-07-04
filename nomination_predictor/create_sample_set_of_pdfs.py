"""
Script for creating a representative sample of PDF files from year-based directories.

This script scans a source directory for PDF files organized in year-named subdirectories,
randomly samples a specified number of files from each year, and copies them to a
destination directory. This is useful for creating balanced datasets for ML training Google's Document AI.
"""

from collections import defaultdict
import logging
from pathlib import Path
import random
import shutil
from typing import Dict, List, Tuple

# --- Configuration ---
# Directory settings
SOURCE_DIRECTORY = Path("tests/fixtures/pages").resolve()
DESTINATION_DIRECTORY = Path("data/raw/pdf_samples").resolve()

# Sampling settings
SAMPLES_PER_YEAR = 2  # Number of random samples per year

# Logging settings
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILE = "pdf_sampling.log"
# ---------------------

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def ensure_directory_exists(directory: Path) -> None:
    """Ensure the specified directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory to check/create.
        
    Raises:
        OSError: If directory creation fails.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", directory)
    except OSError as e:
        logger.error("Failed to create directory %s: %s", directory, e)
        raise


def group_pdfs_by_year(directory: Path) -> Dict[int, List[Path]]:
    """Group PDF files by their parent directory name (year).
    
    Args:
        directory: Directory to search for PDF files.
        
    Returns:
        Dictionary mapping years to lists of PDF file paths.
    """
    pdfs_by_year: Dict[int, List[Path]] = defaultdict(list)
    
    for pdf_file in directory.rglob("*.pdf"):
        try:
            year = int(pdf_file.parent.name)
            pdfs_by_year[year].append(pdf_file)
        except (ValueError, IndexError):
            logger.debug("Skipping file not in year-named directory: %s", pdf_file)
            continue
            
    return pdfs_by_year


def sample_files(files: List[Path], sample_size: int) -> List[Path]:
    """Randomly sample files from a list.
    
    Args:
        files: List of file paths to sample from.
        sample_size: Number of files to sample.
        
    Returns:
        List of sampled file paths.
    """
    if not files:
        return []
    sample_size = min(sample_size, len(files))
    return random.sample(files, sample_size)


def copy_sample_files(
    pdfs_by_year: Dict[int, List[Path]],
    dest_dir: Path,
    samples_per_year: int,
) -> Tuple[int, Dict[int, int]]:
    """Copy sampled files to destination directory.
    
    Args:
        pdfs_by_year: Dictionary mapping years to PDF file paths.
        dest_dir: Destination directory for copied files.
        samples_per_year: Number of files to sample from each year.
        
    Returns:
        Tuple of (total_copied, year_counts) where:
        - total_copied: Total number of files copied
        - year_counts: Dictionary mapping years to number of files copied
    """
    total_copied = 0
    year_counts: Dict[int, int] = {}
    
    for year, files in sorted(pdfs_by_year.items()):
        sampled_files = sample_files(files, samples_per_year)
        year_counts[year] = len(sampled_files)
        
        logger.info("Selected %d file(s) from %d", year_counts[year], year)
        
        for file_path in sampled_files:
            try:
                shutil.copy(file_path, dest_dir)
                total_copied += 1
            except (OSError, shutil.SameFileError) as e:
                logger.error("Failed to copy %s: %s", file_path.name, e)
    
    return total_copied, year_counts


def create_representative_sample() -> None:
    """Main function to create a representative sample of PDF files."""
    try:
        logger.info("Starting PDF sampling process")
        logger.debug("Source directory: %s", SOURCE_DIRECTORY)
        logger.debug("Destination directory: %s", DESTINATION_DIRECTORY)
        logger.debug("Samples per year: %d", SAMPLES_PER_YEAR)
        
        # Validate input
        if not SOURCE_DIRECTORY.is_dir():
            raise NotADirectoryError(f"Source directory not found: {SOURCE_DIRECTORY}")
            
        if SAMPLES_PER_YEAR < 1:
            raise ValueError("SAMPLES_PER_YEAR must be at least 1")
        
        # Ensure destination directory exists
        ensure_directory_exists(DESTINATION_DIRECTORY)
        
        # Group PDFs by year
        logger.info("Scanning for PDF files in '%s'...", SOURCE_DIRECTORY)
        pdfs_by_year = group_pdfs_by_year(SOURCE_DIRECTORY)
        
        if not pdfs_by_year:
            logger.warning("No PDF files found in year-named subdirectories.")
            return
            
        logger.info("Found PDFs across %d different years.", len(pdfs_by_year))
        
        # Sample and copy files
        total_copied, year_counts = copy_sample_files(
            pdfs_by_year, DESTINATION_DIRECTORY, SAMPLES_PER_YEAR
        )
        
        # Log summary
        logger.info("\nSampling complete!")
        logger.info("Copied %d files in total:", total_copied)
        for year, count in sorted(year_counts.items()):
            logger.info("  - %d: %d file(s)", year, count)
            
    except Exception as e:
        logger.exception("An error occurred during PDF sampling: %s", e)
        raise


if __name__ == "__main__":
    create_representative_sample()
