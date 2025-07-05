"""
Web utilities for fetching and processing web content.

This module provides functionality for making HTTP requests, validating URLs,
and generating archive URLs for the judicial vacancies data.
"""

from datetime import datetime
import logging
import time
from typing import Any, Dict, List

from bs4 import BeautifulSoup
import requests

# Configure logging
logger = logging.getLogger(__name__)


class WebUtilsError(Exception):
    """Base exception for web utilities errors."""
    pass


class ValidationError(WebUtilsError):
    """Raised when URL validation fails."""
    pass


class FetchError(WebUtilsError):
    """Raised when there's an error fetching data."""
    pass


class ParseError(WebUtilsError):
    """Raised when there's an error parsing HTML content."""
    pass


def validate_url(url: str) -> bool:
    """
    Validate that a URL is well-formed and uses an allowed scheme.

    Args:
        url: The URL to validate

    Returns:
        bool: True if the URL is valid, False otherwise
    """
    try:
        result = requests.utils.urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception as e:
        logger.debug(f"URL validation failed for {url}: {e}")
        return False


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
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {last_error}")
        except requests.HTTPError as e:
            status_code = getattr(e.response, "status_code", "unknown")
            last_error = f"HTTP error {status_code} occurred: {e}"
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {last_error}")
            # Don't retry on 4xx errors (except 429 Too Many Requests)
            if isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 429:
                raise  # Re-raise the original HTTPError
        except requests.RequestException as e:
            last_error = f"Request failed: {e}"
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {last_error}")

        # If we get here, an exception occurred
        if attempt < max_retries - 1:
            sleep_time = retry_delay * (1 + attempt)  # Exponential backoff
            logger.debug(f"Retrying in {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        continue

    # If we've exhausted all retries
    error_msg = f"Failed to fetch {url} after {max_retries} attempts. Last error: {last_error}"
    logger.error(error_msg)
    raise FetchError(error_msg)


def generate_or_fetch_archive_urls() -> List[str]:
    """
    Generate or fetch URLs for judicial vacancy archive pages.

    Returns:
        List of archive page URLs from 2009 to the current year (inclusive)
        in the format: https://www.uscourts.gov/.../archive-judicial-vacancies?year=YYYY
    """
    base_url = "https://www.uscourts.gov/data-news/judicial-vacancies/archive-judicial-vacancies"
    current_year = datetime.now().year
    
    # Generate URLs from 2009 to the current year (inclusive)
    start_year = 2009
    urls = [f"{base_url}?year={year}" for year in range(start_year, current_year + 1)]
    
    logger.info(f"Generated {len(urls)} archive URLs from {start_year} to {current_year}")
    return urls


def extract_month_links(html: str) -> List[Dict[str, Any]]:
    """
    Extract month links from a year archive page.

    Args:
        html: HTML content of a year archive page

    Returns:
        List of dictionaries with keys:
        - url: URL to the month's vacancy page
        - month: Name of the month
        - year: None (to be set by the caller)
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        month_links = []
        
        # Find the main content area
        content = soup.find('main') or soup.find('div', class_='layout-content') or soup
        
        # Look for the table containing month links
        table = content.find('table', class_='usa-table')
        if not table:
            raise ParseError("Could not find the months table in the HTML")
        
        # Find all table rows (skip header row)
        rows = table.find('tbody').find_all('tr')
        
        # Define valid month names for validation
        valid_months = [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        ]
        
        for row in rows:
            # Get the month cell (first cell in the row)
            month_cell = row.find('td')
            if not month_cell:
                continue
                
            # Get and clean the month name
            month_name = month_cell.get_text(strip=True).lower()
            
            # Validate month name
            if month_name not in valid_months:
                continue
                
            # Find the first link in the second cell (vacancies link)
            links_cell = month_cell.find_next_sibling('td')
            if not links_cell:
                continue
                
            # Look for the vacancies link (first link in the cell)
            link = links_cell.find('a', href=True)
            if not link:
                continue
                
            # Get and clean the URL
            url = link['href'].strip()
            if not url.startswith(('http://', 'https://')):
                if not url.startswith('/'):
                    url = f'/{url}'
            
            month_links.append({
                'url': url,
                'month': month_name.capitalize(),
                'year': None  # Will be set by the caller
            })
        
        return month_links

    except Exception as e:
        raise ParseError(f"Error extracting month links: {e}") from e
