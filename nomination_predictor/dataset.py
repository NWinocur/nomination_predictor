"""
Data pipeline for scraping and processing judicial vacancy data.

This module provides functionality to fetch, parse, and process judicial vacancy data from the US Courts website. It includes functions for web scraping, HTML parsing, and file I/O operations.

This module can refer to config.py to determine where to retrieve data from, and which folder is being used as the raw data folder in which to store data.

This module is NOT meant as a location for code for data transformations.  Transformations should be done in other modules such as features.py.
This module shall deliver dataframes which represent their original website sources (tables on HTML pages and PDFs) as accurately and unchanged as feasible, leaving the work of data cleaning or feature creation to other code.

"""

from datetime import datetime
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Union

from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import requests
from tqdm import tqdm
import typer

from nomination_predictor.config import PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("data_pipeline.log")],
)
logger = logging.getLogger(__name__)  # noqa: F811

# Load environment variables
load_dotenv(find_dotenv())

app = typer.Typer()


# Custom Exceptions
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
    try:
        result = requests.utils.urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def generate_or_fetch_archive_urls() -> List[str]:
    """
    Generate or fetch URLs for judicial vacancy archive pages.

    Returns:
        List of archive page URLs from 1981 to the current year (inclusive)
        in the format: https://www.uscourts.gov/.../archive-judicial-vacancies?year=YYYY
    """
    base_url = "https://www.uscourts.gov/data-news/judicial-vacancies/archive-judicial-vacancies"
    current_year = datetime.now().year
    
    # Generate URLs from 1981 to the current year (inclusive)
    start_year = 1981
    urls = [f"{base_url}?year={year}" for year in range(start_year, current_year + 1)]
    
    return urls


def fetch_html(url: str, max_retries: int = 3, retry_delay: float = 1.0, timeout: int = 30) -> str:
    """
    Fetch HTML content from a URL with retries and error handling.

    Args:
        url: URL to fetch
        max_retries: Maximum number of retry attempts (must be >= 1)
        retry_delay: Delay between retries in seconds
        timeout: Request timeout in seconds (default: 30)

    Returns:
        str: HTML content as string

    Raises:
        ValueError: If max_retries is not a positive integer
        ValidationError: If the URL is invalid
        FetchError: If the request fails after all retries

    Example:
        >>> html = fetch_html("https://example.com")
    """
    # Validate max_retries
    if not isinstance(max_retries, int) or max_retries < 1:
        raise ValueError("max_retries must be a positive integer")

    # Validate URL
    if not validate_url(url):
        raise ValidationError(f"Invalid URL: {url}")

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text

        except requests.Timeout as e:
            last_error = f"Request timed out: {e}"
        except requests.HTTPError as e:
            status_code = getattr(e.response, "status_code", "unknown")
            last_error = f"HTTP error {status_code} occurred: {e}"
            # Don't retry on 4xx errors (except 429 Too Many Requests)
            if isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 429:
                break
        except requests.RequestException as e:
            last_error = f"Request failed: {e}"

        # If we get here, an exception occurred
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (1 + attempt))  # Exponential backoff
        continue

    # If we've exhausted all retries
    raise FetchError(
        f"Failed to fetch {url} after {max_retries} attempts. Last error: {last_error}"
    )


def extract_vacancy_table(html: str) -> List[Dict[str, Any]]:
    """
    Extract judicial vacancy data from month-level HTML content.

    Args:
        html: HTML content containing a table with judicial vacancy data

    Returns:
        List of dictionaries containing extracted data with standardized fields:
        - court: Name of the court/district (required)
        - vacancy_date: Date the vacancy occurred (required)
        - incumbent: Name of the outgoing judge (required)
        - vacancy_reason: Reason for the vacancy (required)
        - nominee: Name of the nominee (if any)
        - nomination_date: Date of nomination (if any)
        - confirmation_date: Date of confirmation (if any)
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return []

        # Get headers and normalize them (lowercase, replace spaces with underscores)
        headers = []
        header_row = table.find("tr")
        if header_row:
            headers = [
                th.get_text(strip=True).lower().replace(" ", "_")
                for th in header_row.find_all(["th", "td"])
            ]
        else:
            # If no header row, use default column names based on position
            headers = [
                "court", 
                "vacancy_date", 
                "incumbent", 
                "vacancy_reason", 
                "nominee", 
                "nomination_date", 
                "confirmation_date"
            ]

        rows = []
        # Skip header row if it exists
        for row in table.find_all("tr")[1 if header_row else 0:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if not cells:  # Skip empty rows
                continue

            # Create record with fields from the row
            record = {}
            for i, value in enumerate(cells):
                if i < len(headers):
                    field_name = headers[i]
                    # Only include non-empty values
                    if value and value.strip():
                        record[field_name] = value.strip()

            # Only add the record if it has required fields
            if all(field in record for field in ["court", "vacancy_date"]):
                rows.append(record)

        return rows

    except Exception as e:
        raise ParseError(f"Failed to parse HTML table: {e}")


def extract_month_links(html: str, base_url: str = "https://www.uscourts.gov") -> List[Dict[str, str]]:
    """
    Extract month-level links from a year page.
    
    Args:
        html: HTML content of the year page
        base_url: Base URL to resolve relative links (default: uscourts.gov)
        
    Returns:
        List of dictionaries with 'url' and 'month' keys
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        month_links = []
        
        # Find all links that match the pattern for monthly reports
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True).lower()
            
            # Look for links that match the pattern for monthly reports
            if '/judicial-vacancies/' in href and any(month in text for month in [
                'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december'
            ]):
                # Resolve relative URLs
                if not href.startswith(('http://', 'https://')):
                    href = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                
                month_links.append({
                    'url': href,
                    'month': text,
                    'year': None  # Will be set by the caller
                })
        
        return month_links
        
    except Exception as e:
        logger.error(f"Error extracting month links: {e}")
        raise ParseError(f"Failed to extract month links: {e}")


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of records to a pandas DataFrame.

    Args:
        records: List of dictionaries containing the data

    Returns:
        DataFrame containing the processed data
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Convert numeric columns
    if "seat" in df.columns:
        df["seat"] = pd.to_numeric(df["seat"], errors="coerce")

    # Convert date columns
    date_columns = ["vacancy_date", "nomination_date"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def save_to_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df: DataFrame to save
        path: Output file path

    Raises:
        DataPipelineError: If saving fails
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Successfully saved data to {path}")
    except Exception as e:
        raise DataPipelineError(f"Failed to save CSV to {path}: {e}")


@app.command()
def main(output_dir: Path = PROCESSED_DATA_DIR, output_filename: str = "judicial_vacancies.csv"):
    """
    Main entry point for the data pipeline.

    Args:
        output_dir: Directory to save output files
        output_filename: Name of the output CSV file
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        logger.info("Starting data pipeline...")

        # Fetch and process data
        urls = generate_or_fetch_archive_urls()
        all_records = []

        for url in tqdm(urls, desc="Processing archive pages"):
            try:
                html = fetch_html(url)
                records = extract_vacancy_table(html)
                all_records.extend(records)
                logger.debug(f"Processed {len(records)} records from {url}")
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                continue

        # Convert to DataFrame and save
        if all_records:
            df = records_to_dataframe(all_records)
            save_to_csv(df, output_path)
            logger.success(f"Successfully processed {len(df)} records to {output_path}")
        else:
            logger.warning("No records were processed")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    app()
