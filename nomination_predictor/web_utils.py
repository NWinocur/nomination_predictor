"""
Web utilities for fetching and processing web content.

This module provides functionality for making HTTP requests, validating URLs,
and generating archive URLs for the judicial vacancies data.
"""

from datetime import datetime
import logging
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

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


@patch("requests.get")
def test_fetch_html_success(mock_get):
    """Test successful HTML fetching."""
    # Mock the response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html>Test</html>"
    mock_get.return_value = mock_response

    # Test the function
    result = fetch_html("http://example.com")
    assert result == "<html>Test</html>"
    mock_get.assert_called_once_with("http://example.com", timeout=60)


@patch("requests.get")
def test_fetch_html_retry_on_timeout(mock_get):
    """Test fetch_html retries on timeout."""
    # Mock responses: timeout once, then succeed
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html>Success</html>"

    mock_get.side_effect = [requests.exceptions.Timeout("Connection timed out"), mock_response]

    result = fetch_html("http://example.com", max_retries=2, retry_delay=0.1)
    assert result == "<html>Success</html>"
    assert mock_get.call_count == 2


@patch("requests.get")
def test_fetch_html_retry_on_connection_error(mock_get):
    """Test fetch_html retries on connection error."""
    # Mock responses: connection error once, then succeed
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html>Success</html>"

    mock_get.side_effect = [
        requests.exceptions.ConnectionError("Connection failed"),
        mock_response,
    ]

    result = fetch_html("http://example.com", max_retries=2, retry_delay=0.1)
    assert result == "<html>Success</html>"
    assert mock_get.call_count == 2





@patch("requests.get")
def test_fetch_html_http_error_5xx(mock_get):
    """Test fetch_html retries on 5xx errors."""
    # Mock a 500 response that fails all retries
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    mock_get.return_value = mock_response

    with pytest.raises(FetchError, match="Failed to fetch"):
        fetch_html("http://example.com/error", max_retries=2)

    assert mock_get.call_count == 2


@patch("requests.get")
def test_fetch_html_rate_limit(mock_get):
    """Test fetch_html handles 429 Too Many Requests with retry."""
    # Mock a 429 response followed by success
    error_response = MagicMock()
    error_response.status_code = 429
    error_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.text = "<html>Success</html>"

    mock_get.side_effect = [error_response, success_response]

    result = fetch_html("http://example.com", max_retries=2, retry_delay=0.1)
    assert result == "<html>Success</html>"
    assert mock_get.call_count == 2


def test_fetch_html_invalid_retries():
    """Test fetch_html validates max_retries parameter."""
    with pytest.raises(ValueError, match="max_retries must be a positive integer"):
        fetch_html("http://example.com", max_retries=0)

    with pytest.raises(ValueError, match="max_retries must be a positive integer"):
        fetch_html("http://example.com", max_retries=-1)

    with pytest.raises(ValueError, match="max_retries must be a positive integer"):
        fetch_html("http://example.com", max_retries="not-an-integer")

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


def fetch_html(url: str, max_retries: int = 2, retry_delay: float = 5.0, timeout: int = 60) -> str:
    """
    Fetch HTML content from a URL with retries and error handling.

    Args:
        url: URL to fetch
        max_retries: Maximum number of retry attempts (must be >= 1, default: 2)
        retry_delay: Base delay between retries in seconds (default: 5.0)
        timeout: Request timeout in seconds (default: 60)

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
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text

        except requests.Timeout as e:
            last_error = f"Request timed out after {timeout}s: {e}"
            logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {last_error}")
        except requests.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else None
            last_error = f"HTTP error {status_code} occurred: {e}"
            logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {last_error}")
            # Don't retry on client errors (4xx) except 429 Too Many Requests
            if status_code and 400 <= status_code < 500 and status_code != 429:
                raise FetchError(f"Client error {status_code}: {e}") from e
        except requests.RequestException as e:
            last_error = f"Request failed: {e}"
            logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {last_error}")
        
        # Don't sleep after the last attempt
        if attempt < max_retries:
            # Exponential backoff: 5s, 10s, 15s, etc.
            delay = retry_delay * (attempt + 1)
            logger.debug(f"Waiting {delay:.1f}s before retry...")
            time.sleep(delay)

    raise FetchError(f"Failed after {max_retries + 1} attempts. Last error: {last_error}")


def generate_or_fetch_archive_urls() -> List[str]:
    """
    Generate or fetch URLs for year-level judicial vacancy archive pages.

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


def generate_month_links(year: int, page_type: str = "vacancies") -> List[Dict[str, Any]]:
    """
    Generate month links for a given year and page type.
    
    The URLs follow a predictable pattern:
    /judges-judgeships/judicial-vacancies/archive-judicial-vacancies/YYYY/MM/{page_type}

    Args:
        year: Year of the archive page (as an integer, e.g., 2025)
        page_type: Type of page to generate links for. Must be one of:
                  - 'vacancies' (default)
                  - 'confirmations'
                  - 'emergencies'

    Returns:
        List of dictionaries with keys:
        - url: URL to the month's page
        - month: two-digit number of the month as a string (single-digit months are zero-padded)
        - year: 4-digit year of the archive page (as a string)
        - page_type: the type of page (same as the input parameter)

    Raises:
        ValueError: If an invalid page_type is provided
        ParseError: If there's an error generating the month links
    """
    # Validate page_type
    valid_page_types = ["vacancies", "confirmations", "emergencies"]
    if page_type not in valid_page_types:
        raise ValueError(
            f"Invalid page_type: {page_type}. Must be one of {valid_page_types}"
        )
    
    try:
        month_links = []
        if year == 2009:
            for i in range(7, 13):
                month_num = f"{i:02d}"  # Zero-pad month number
                url = (
                    f"/judges-judgeships/judicial-vacancies/"
                    f"archive-judicial-vacancies/{year}/{month_num}/{page_type}"
                )
                
                month_links.append({
                    'url': url,
                    'month': month_num,
                    'year': str(year),  # Convert year to string to match test expectations
                    'page_type': page_type
                })
        else:
            for i in range(1, 13):
                month_num = f"{i:02d}"  # Zero-pad month number
                url = (
                    f"/judges-judgeships/judicial-vacancies/"
                    f"archive-judicial-vacancies/{year}/{month_num}/{page_type}"
                )
                
                month_links.append({
                    'url': url,
                    'month': month_num,
                    'year': str(year),  # Convert year to string to match test expectations
                    'page_type': page_type
                })
        
        return month_links
        
    except Exception as e:
        raise ParseError(f"Error generating month links: {e}") from e


def test_generate_or_fetch_archive_urls():
    """
    Test URL generation for archive pages.

    Verifies that:
    1. Returns a list of strings
    2. All URLs are valid HTTP/HTTPS URLs with the correct format
    3. Covers years from 2009 to current year (inclusive)
    4. Uses the correct query parameter format: ?year=YYYY
    """
    # Get the current year
    current_year = datetime.now().year

    # Generate the URLs
    urls = generate_or_fetch_archive_urls()

    # Basic type and format checks
    assert isinstance(urls, list)
    assert all(isinstance(url, str) for url in urls)
    assert all(url.startswith(("http://", "https://")) for url in urls)

    # Check year range (2009 to current year, inclusive)
    years = []
    for url in urls:
        # Extract the year from the URL query parameter
        try:
            from urllib.parse import parse_qs, urlparse

            parsed_url = urlparse(url)
            year = int(parse_qs(parsed_url.query).get("year", [""])[0])
            years.append(year)

            # Verify the URL format
            assert parsed_url.path.endswith("/archive-judicial-vacancies"), (
                "URL path should end with '/archive-judicial-vacancies'"
            )
            assert "year=" in parsed_url.query, "URL should contain 'year' query parameter"

        except (ValueError, IndexError, AssertionError) as e:
            pytest.fail(f"Invalid URL format: {url}. Error: {e}")

    # Should include all years from 2009 to current year
    expected_years = list(range(2009, current_year + 1))
    assert sorted(years) == expected_years, (
        f"Expected years {expected_years[0]}-{expected_years[-1]}, got {min(years)}-{max(years)}"
    )

    # Verify URL format for a sample year
    sample_year = 2020
    assert any(f"year={sample_year}" in url for url in urls), (
        f"Expected to find URL with '?year={sample_year}'"
    )

