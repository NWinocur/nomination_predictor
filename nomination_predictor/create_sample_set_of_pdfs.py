"""
Script for creating a representative sample of PDF files from year-based directories.

This script scans a source directory for specific PDF files (confirmation.pdf, vacancies.pdf, 
emergencies.pdf) organized in year-named subdirectories. It creates a balanced sample across
these categories while preserving the directory structure to prevent filename collisions.
"""

from collections import defaultdict, namedtuple
import logging
from pathlib import Path
import random
import shutil
from typing import DefaultDict, Dict, List

# --- Configuration ---
# Directory settings
SOURCE_DIRECTORY = Path("tests/fixtures/pages").resolve()
DESTINATION_DIRECTORY = Path("data/external/pdf_samples").resolve()

# File patterns to include (case-insensitive)
TARGET_FILES = {"confirmations.pdf", "vacancies.pdf", "emergencies.pdf"}

# Sampling settings
SAMPLES_PER_CATEGORY = 25  # Target number of samples per category
SAMPLES_PER_CATEGORY_PER_YEAR = 2  # Number of samples per category per year

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

# Named tuple to track file information
FileInfo = namedtuple('FileInfo', ['source_path', 'relative_path', 'year', 'category'])


def ensure_directory_exists(directory: Path) -> None:
    """Ensure the specified directory exists, creating it if necessary."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", directory)
    except OSError as e:
        logger.error("Failed to create directory %s: %s", directory, e)
        raise


def get_relative_path(source_root: Path, file_path: Path) -> Path:
    """Get the relative path of a file from the source root."""
    try:
        return file_path.relative_to(source_root.parent)
    except ValueError:
        return file_path.name


def find_target_files(directory: Path) -> Dict[str, List[FileInfo]]:
    """Find all target PDF files, organized by category."""
    files_by_category: DefaultDict[str, List[FileInfo]] = defaultdict(list)
    
    for pdf_file in directory.rglob("*.pdf"):
        filename = pdf_file.name.lower()
        if filename not in TARGET_FILES:
            continue
            
        try:
            year = int(pdf_file.parent.name)
            relative_path = get_relative_path(directory, pdf_file)
            file_info = FileInfo(
                source_path=pdf_file,
                relative_path=relative_path,
                year=year,
                category=filename
            )
            files_by_category[filename].append(file_info)
        except (ValueError, IndexError):
            logger.debug("Skipping file not in year-named directory: %s", pdf_file)
            
    return files_by_category


def sample_files(
    files_by_category: Dict[str, List[FileInfo]],
    samples_per_category: int,
    samples_per_category_per_year: int = 2
) -> List[FileInfo]:
    """Sample files ensuring representation from each category and year.
    
    First takes up to samples_per_category_per_year from each year for each category,
    then fills remaining samples from any available files if needed to reach samples_per_category.
    """
    sampled_files: List[FileInfo] = []
    
    for category, files in files_by_category.items():
        if not files:
            logger.warning("No files found for category: %s", category)
            continue
            
        # First, group files by year for this category
        files_by_year: Dict[int, List[FileInfo]] = defaultdict(list)
        for file_info in files:
            files_by_year[file_info.year].append(file_info)
        
        # Sample up to samples_per_category_per_year from each year
        yearly_sampled: List[FileInfo] = []
        remaining_files: List[FileInfo] = []
        
        for year, year_files in files_by_year.items():
            # Take up to samples_per_category_per_year from this year
            sample_size = min(samples_per_category_per_year, len(year_files))
            sampled = random.sample(year_files, sample_size)
            yearly_sampled.extend(sampled)
            
            # Keep track of remaining files for potential additional sampling
            remaining_files.extend([f for f in year_files if f not in sampled])
        
        # If we don't have enough samples from the per-year sampling,
        # take more from the remaining files
        if len(yearly_sampled) < samples_per_category and remaining_files:
            needed = samples_per_category - len(yearly_sampled)
            needed = min(needed, len(remaining_files))
            additional_samples = random.sample(remaining_files, needed)
            yearly_sampled.extend(additional_samples)
        
        # Limit to the requested sample size (in case we have more than needed)
        if len(yearly_sampled) > samples_per_category:
            yearly_sampled = random.sample(yearly_sampled, samples_per_category)
        
        sampled_files.extend(yearly_sampled)
        logger.info("Selected %d %s files", len(yearly_sampled), category)
    
    return sampled_files


def copy_files(
    files: List[FileInfo],
    dest_dir: Path,
) -> int:
    """Copy files to destination, preserving directory structure."""
    total_copied = 0
    
    for file_info in files:
        dest_path = dest_dir / file_info.relative_path
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_info.source_path, dest_path)
            total_copied += 1
            logger.debug("Copied %s to %s", file_info.source_path, dest_path)
        except (OSError, shutil.SameFileError) as e:
            logger.error("Failed to copy %s: %s", file_info.source_path, e)
    
    return total_copied


def create_representative_sample() -> None:
    """Main function to create a balanced sample of PDF files across categories."""
    try:
        logger.info("Starting PDF sampling process")
        logger.debug("Source directory: %s", SOURCE_DIRECTORY)
        logger.debug("Destination directory: %s", DESTINATION_DIRECTORY)
        logger.debug("Target files: %s", ", ".join(sorted(TARGET_FILES)))
        
        if not SOURCE_DIRECTORY.is_dir():
            raise NotADirectoryError(f"Source directory not found: {SOURCE_DIRECTORY}")
        
        ensure_directory_exists(DESTINATION_DIRECTORY)
        
        # Find all target files
        logger.info("Scanning for target PDF files in '%s'...", SOURCE_DIRECTORY)
        files_by_category = find_target_files(SOURCE_DIRECTORY)
        
        # Log found files by category
        for category in sorted(TARGET_FILES):
            count = len(files_by_category.get(category, []))
            logger.info("Found %d %s files", count, category)
        
        if not any(files_by_category.values()):
            logger.warning("No target PDF files found in subdirectories.")
            return
        
        # Sample files
        logger.info("Sampling files...")
        sampled_files = sample_files(
            files_by_category,
            samples_per_category=SAMPLES_PER_CATEGORY,
            samples_per_category_per_year=SAMPLES_PER_CATEGORY_PER_YEAR
        )
        
        # Copy files
        logger.info("Copying files to destination...")
        total_copied = copy_files(sampled_files, DESTINATION_DIRECTORY)
        
        # Log summary
        logger.info("\nSampling complete!")
        logger.info("Copied %d files in total:", total_copied)
        
        # Count by category and year
        by_category: DefaultDict[str, int] = defaultdict(int)
        by_year: DefaultDict[int, int] = defaultdict(int)
        
        for file_info in sampled_files:
            by_category[file_info.category] += 1
            by_year[file_info.year] += 1
        
        logger.info("By category:")
        for category in sorted(by_category):
            logger.info("  - %s: %d files", category, by_category[category])
            
        logger.info("By year:")
        for year in sorted(by_year):
            logger.info("  - %d: %d files", year, by_year[year])
            
    except Exception as e:
        logger.exception("An error occurred during PDF sampling: %s", e)
        raise


if __name__ == "__main__":
    create_representative_sample()
