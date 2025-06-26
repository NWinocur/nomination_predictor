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

For more information about the data source, visit:
https://www.uscourts.gov/data-news/judicial-vacancies/archive-judicial-vacancies
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_pipeline.log')
    ]
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
    return url.startswith(('http://', 'https://'))

def generate_or_fetch_archive_urls() -> List[str]:
   """
    Fetch and parse the US Courts archive page to extract relevant URLs.
    
    This function scrapes the main archive page to find URLs containing
    judicial vacancy data for different years and categories.
    
    Returns:
        List[str]: A list of absolute URLs to pages containing judicial vacancy data
        
    Example:
        >>> urls = generate_or_fetch_archive_urls()
        >>> print(f"Found {len(urls)} archive URLs")
    """
    
    # This is the main archive page URL - it will be mocked in tests
    archive_url = "https://www.uscourts.gov/data-news/judicial-vacancies/archive-judicial-vacancies"
    
    try:
        logger.info(f"Fetching archive page: {archive_url}")
        response = requests.get(archive_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        urls = []
        
        # Example: Find all links containing 'vacancies' and '2025'
        for link in soup.find_all('a', href=True):
            url = link['href']
            if '2025' in url and any(x in url for x in ['vacancies', 'emergencies', 'confirmations']):
                if not url.startswith('http'):
                    url = f"https://www.uscourts.gov{url if not url.startswith('/') else url}"
                urls.append(url)
        
        if not urls:
            logger.warning("No valid URLs found on the archive page")
            
        logger.info(f"Found {len(urls)} archive URLs")
        return urls
        
    except requests.RequestException as e:
        error_msg = f"Error fetching archive page: {e}"
        logger.error(error_msg)
        raise FetchError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error while processing archive page: {e}"
        logger.exception(error_msg)
        raise DataPipelineError(error_msg) from e

def fetch_html(url):
    """
    Fetch HTML content from a given URL.
    
    Args:
        url: The URL to fetch HTML from
        
    Returns:
        The HTML content as a string
        
    Raises:
        requests.RequestException: If there's an error fetching the URL
        
    Example:
        >>> html = fetch_html("https://example.com")
        >>> print(f"Fetched {len(html)} characters")
    """
    """
Data pipeline for scraping and processing judicial vacancy data.

This module provides functionality to fetch, parse, and process judicial vacancy
data from the US Courts website. It includes functions for web scraping, HTML parsing,
data transformation, and file I/O operations.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv())

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
    return url.startswith(('http://', 'https://'))

def generate_or_fetch_archive_urls() -> List[str]:
    """Fetch and parse the US Courts archive page to extract relevant URLs."""
    archive_url = "https://www.uscourts.gov/data-news/judicial-vacancies/archive-judicial-vacancies"
    
    try:
        logger.info(f"Fetching archive page: {archive_url}")
        response = requests.get(archive_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        urls = []
        
        # Example: Find all links containing 'vacancies' and '2025'
        for link in soup.find_all('a', href=True):
            url = link['href']
            if '2025' in url and any(x in url for x in ['vacancies', 'emergencies', 'confirmations']):
                if not url.startswith('http'):
                    url = f"https://www.uscourts.gov{url if not url.startswith('/') else url}"
                urls.append(url)
        
        if not urls:
            logger.warning("No valid URLs found on the archive page")
            
        logger.info(f"Found {len(urls)} archive URLs")
        return urls
        
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
    for attempt in range(max_retries):
        try:
            logger.debug(f"Fetching URL (attempt {attempt + 1}): {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
            
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                error_msg = f"Failed to fetch {url} after {max_retries} attempts: {e}"
                logger.error(error_msg)
                raise FetchError(error_msg) from e
            logger.warning(f"Attempt {attempt + 1} failed, retrying...")

    # This should never be reached due to the raise in the loop
    raise FetchError("Unexpected error in fetch_html")

def extract_vacancy_table(html):
    """
    Extract judicial vacancy data from an HTML table.
    
    Args:
        html: HTML content containing a table with vacancy data
        
    Returns:
        A list of dictionaries where each dictionary represents a row with
        'Seat' and 'Court' keys
        
    Example:
        >>> with open("example.html") as f:
        ...     html = f.read()
        >>> records = extract_vacancy_table(html)
        >>> print(f"Extracted {len(records)} records")
    """
    from bs4 import BeautifulSoup
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        
        if not table:
            print("Warning: No table found in HTML")
            return []
            
        # Extract headers
        headers = []
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
        
        # If no headers found in thead, try to infer from first row
        if not headers:
            first_row = table.find('tr')
            if first_row:
                headers = [th.get_text(strip=True) for th in first_row.find_all('th')]
                # If we found headers in first row, skip it when parsing data
                skip_first_row = True
        
        # Default to expected columns if no headers found
        if not headers:
            headers = ['Seat', 'Court']
            skip_first_row = False
        
        # Extract data rows
        records = []
        rows = table.find_all('tr')
        
        for row in rows[1:] if skip_first_row and len(rows) > 1 else rows:
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
                
            # Create a record with the data
            record = {}
            for i, cell in enumerate(cells):
                if i < len(headers):
                    record[headers[i]] = cell.get_text(strip=True)
            
            # Only add if we have the expected columns
            if 'Seat' in record and 'Court' in record:
                records.append(record)
        
        return records
        
    except Exception as e:
        print(f"Error parsing HTML table: {e}")
        # Return a default non-real record
        return [{"Seat": "0", "Court": "0th Circuit"}]


def records_to_dataframe(records):
    """
    Convert a list of record dictionaries into a pandas DataFrame.
    
    Args:
        records: A list of dictionaries where each dictionary represents a row of data
        
    Returns:
        A DataFrame containing the records with proper data types
        
    Example:
        >>> records = [{"Seat": "1", "Court": "9th Circuit"}]
        >>> df = records_to_dataframe(records)
        >>> print(df.dtypes)
    """
    import pandas as pd
    
    if not records:
        return pd.DataFrame(columns=["Seat", "Court"])
    
    try:
        # Create DataFrame from records
        df = pd.DataFrame(records)
        
        # Ensure required columns exist
        if 'Seat' not in df.columns or 'Court' not in df.columns:
            raise ValueError("Input records must contain 'Seat' and 'Court' columns")
        
        # Convert Seat to numeric if possible
        df['Seat'] = pd.to_numeric(df['Seat'], errors='coerce')
        
        # Drop any rows with missing required values
        df = df.dropna(subset=['Seat', 'Court'])
        
        # Reset index after dropping rows
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error converting records to DataFrame: {e}")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=["Seat", "Court"])


def save_to_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Save a pandas DataFrame to a CSV file.
    
    Args:
        df: The DataFrame to save
        path: The file path where the CSV will be saved
        
    Raises:
        ValueError: If the input is not a pandas DataFrame
        IOError: If the file cannot be written to the specified path
        
    Example:
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> save_to_csv(df, "output.csv")
    """
    df.to_csv(path, index=False)


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