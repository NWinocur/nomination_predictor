"""
Data pipeline for scraping and processing judicial vacancy data.

This module provides functionality to fetch, parse, and process judicial vacancy
data from the US Courts website. It includes functions for web scraping, HTML parsing,
data transformation, and file I/O operations.

    >>> from data import make_dataset
    >>> urls = make_dataset.generate_or_fetch_archive_urls()
    >>> html = make_dataset.fetch_html(urls[0])
    >>> records = make_dataset.extract_vacancy_table(html)
    >>> df = make_dataset.records_to_dataframe(records)
    >>> make_dataset.save_to_csv(df, "judicial_vacancies.csv")
    
Raw CSV files are then saved to the project's `data/raw` directory.

For more information about the data source, visit:
https://www.uscourts.gov/data-news/judicial-vacancies/archive-judicial-vacancies
"""

from datetime import datetime
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union

from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import requests

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("data_pipeline.log")],
)
logger = logging.getLogger(__name__)

# load up the entries as environment variables
load_dotenv(dotenv_path)

database_url = os.environ.get("DATABASE_URL")
other_variable = os.environ.get("OTHER_VARIABLE")


class DataPipelineError(Exception):
    """Base exception for data pipeline errors."""

    pass


class FetchError(DataPipelineError):
    """Raised when there's an error fetching data."""

    pass


class ParseError(DataPipelineError):
    """Raised when there's an error parsing data."""

    pass


class ValidationError(DataPipelineError):
    """Raised when data validation fails."""

    pass


def validate_url(url: str) -> bool:
    """Validate that a URL is well-formed and uses an allowed scheme."""
    return url.startswith(("http://", "https://"))


def generate_or_fetch_archive_urls() -> List[str]:
    """Fetch and parse the US Courts archive page to extract relevant URLs."""
    archive_url = (
        "https://www.uscourts.gov/data-news/judicial-vacancies/archive-judicial-vacancies"
    )

    try:
        logger.info(f"Fetching archive page: {archive_url}")
        response = requests.get(archive_url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        urls = []

        # Look for links containing any year and our target patterns
        for link in soup.find_all("a", href=True):
            url = link["href"]
            # Match any 4-digit year and our target patterns
            if any(x in url for x in ["vacancies", "emergencies", "confirmations"]) and any(
                str(year) in url for year in range(1981, datetime.now().year + 1)
            ):
                if not url.startswith("http"):
                    url = f"https://www.uscourts.gov{url if not url.startswith('/') else url}"
                urls.append(url)

        if not urls:
            logger.warning("No valid URLs found on the archive page")
            raise FetchError("No valid URLs found on the archive page")

        logger.info(f"Found {len(urls)} archive URLs")
        return list(dict.fromkeys(urls))  # Remove duplicates while preserving order

    except requests.RequestException as e:
        error_msg = f"Error fetching archive page: {e}"
        logger.error(error_msg)
        raise FetchError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error while processing archive page: {e}"
        logger.exception(error_msg)
        raise DataPipelineError(error_msg) from e


def fetch_html(url: str) -> str:
    """Fetch HTML content from a given URL with retries and timeout."""
    if not validate_url(url):
        error_msg = f"Invalid URL: {url}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    max_retries = 3
    last_exception = None

    for attempt in range(max_retries):
        try:
            logger.debug(f"Fetching URL (attempt {attempt + 1}): {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # This will raise HTTPError for 4XX/5XX responses
            return response.text

        except requests.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:  # Don't log on the last attempt
                retry_delay = (2**attempt) * 0.1  # Exponential backoff
                logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay:.1f}s..."
                )
                time.sleep(retry_delay)
            continue

    # If we've exhausted all retries
    error_msg = f"Failed to fetch {url} after {max_retries} attempts: {str(last_exception)}"
    logger.error(error_msg)
    raise FetchError(error_msg) from last_exception


def extract_vacancy_table(html: str) -> List[Dict[str, str]]:
    """
    Extract judicial vacancy data from an HTML table.

    Args:
        html: HTML content containing a table with vacancy data

    Returns:
        A list of dictionaries where each dictionary represents a row with
        normalized field names (lowercase, spaces replaced with underscores)

    Example:
        >>> with open("example.html") as f:
        ...     html = f.read()
        >>> records = extract_vacancy_table(html)
        >>> print(f"Extracted {len(records)} records")
    """
    from bs4 import BeautifulSoup

    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        if not table:
            logger.warning("No table found in HTML content")
            return []

        # Extract and normalize headers from thead if it exists
        headers = []
        thead = table.find("thead")
        if thead:
            header_row = thead.find("tr")
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

        # If no headers in thead, try to get from first row in tbody
        if not headers:
            first_row = table.find("tr")  # Look for first row in table
            if first_row:
                headers = [th.get_text(strip=True) for th in first_row.find_all(["th", "td"])]

        if not headers:
            logger.warning("No headers found in table")
            return []

        # Normalize header names (lowercase, replace spaces with underscores)
        normalized_headers = [h.lower().replace(" ", "_") for h in headers]
        logger.debug(f"Found headers: {normalized_headers}")

        # Extract data rows from tbody, or from table if no tbody
        tbody = table.find("tbody")
        rows = tbody.find_all("tr") if tbody else table.find_all("tr")
        
        # If we found headers in thead, we can use all rows in tbody
        # Otherwise, skip the first row since it was used for headers
        start_idx = 0 if thead else 1
        
        logger.debug(f"Found {len(rows)} total rows (starting from index {start_idx})")

        records = []
        for i, row in enumerate(rows[start_idx:], start=start_idx):
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            # Only process rows with the expected number of cells
            if len(cells) == len(normalized_headers):
                record = {}
                for j, cell in enumerate(cells):
                    field_name = normalized_headers[j]
                    record[field_name] = cell.get_text(strip=True)
                if record:  # Only add non-empty records
                    records.append(record)
                    logger.debug(f"Added record: {record}")
            else:
                logger.warning(
                    f"Skipping row with {len(cells)} cells (expected {len(normalized_headers)}): "
                    f"{[cell.get_text(strip=True) for cell in cells]}"
                )

        logger.info(f"Extracted {len(records)} records from HTML table")
        return records

    except Exception as e:
        error_msg = f"Error extracting data from HTML table: {e}"
        logger.error(error_msg)
        raise ParseError(error_msg) from e


def records_to_dataframe(records):
    """
    Convert a list of record dictionaries into a pandas DataFrame.

    Args:
        records: A list of dictionaries where each dictionary represents a row of data

    Returns:
        A DataFrame containing the records with proper data types. Returns an empty
        DataFrame with no columns if the input list is empty.

    Example:
        >>> records = [{"Seat": "1", "Court": "9th Circuit"}]
        >>> df = records_to_dataframe(records)
        >>> print(df.dtypes)
    """
    import pandas as pd

    if not records:
        return pd.DataFrame()  # Return empty DataFrame with no columns

    try:
        # Create DataFrame from records
        df = pd.DataFrame(records)

        # Check for required columns
        required_columns = {"Seat", "Court"}
        if not required_columns.issubset(df.columns):
            logger.warning(f"Missing required columns. Expected at least: {required_columns}")
            return pd.DataFrame()

        # Convert Seat to numeric if possible
        df["Seat"] = pd.to_numeric(df["Seat"], errors="coerce")

        # Drop any rows with missing required values
        df = df.dropna(subset=["Seat", "Court"])

        # Reset index after dropping rows
        df = df.reset_index(drop=True)

        return df

    except Exception as e:
        error_msg = f"Error converting records to DataFrame: {e}"
        logger.error(error_msg)
        # Return an empty DataFrame with no columns on error
        return pd.DataFrame()


def save_to_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        df: The DataFrame to save
        path: The file path where the CSV will be saved

    Raises:
        TypeError: If path is not a string or Path object
        ValueError: If the input is not a pandas DataFrame
        IOError: If the file cannot be written to the specified path

    Example:
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> save_to_csv(df, "output.csv")
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        error_msg = f"Expected pandas DataFrame, got {type(df).__name__}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate path type
    if not isinstance(path, (str, Path)):
        error_msg = f"path must be a string or Path, got {type(path).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    try:
        # Convert to Path object if it's a string
        path = Path(path) if isinstance(path, str) else path
        
        # Create parent directories if they don't exist
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            error_msg = f"Failed to create directory {path.parent}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
            
        # Save the DataFrame to CSV
        try:
            df.to_csv(path, index=False)
            logger.info(f"Successfully saved data to {path}")
        except (OSError, IOError) as e:
            error_msg = f"Error saving DataFrame to {path}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
            
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"Unexpected error when saving to {path}: {e}"
        logger.error(error_msg)
        raise IOError(error_msg) from e


def main() -> None:
    """
    Main function to run the data pipeline.

    This function orchestrates the entire data pipeline:
    1. Fetches archive URLs
    2. Downloads and parses HTML content
    3. Extracts and processes vacancy data
    4. Saves the results to a CSV file

    Example:
        >>> from data import make_dataset
        >>> make_dataset.main()
    """
    # Step 1: Determine which pages to scrape (static list or dynamic scraping)
    urls = generate_or_fetch_archive_urls()

    all_records = []

    for url in urls:
        # Step 2: Fetch and parse HTML from the URL
        html = fetch_html(url)

        # Step 3: Extract and clean tables from HTML
        records = extract_vacancy_table(html)

        # Step 4: Append to master list
        all_records.extend(records)

    # Step 5: Convert to DataFrame
    df = records_to_dataframe(all_records)

    # Step 6: Save to CSV
    save_to_csv(df, "data/raw/judicial_vacancies.csv")


if __name__ == "__main__":
    main()
