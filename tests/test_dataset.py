"""Tests for the dataset module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

# Import the module to test
from nomination_predictor.dataset import (
    DataPipelineError,
    extract_vacancy_table,
    fetch_html,
    generate_or_fetch_archive_urls,
    records_to_dataframe,
    save_to_csv,
)
from nomination_predictor.web_utils import FetchError


# Fixtures
@pytest.fixture
def real_vacancy_html():
    """Return the content of a real vacancies page from the fixtures."""
    path = Path(__file__).parent / "fixtures" / "pages" / "2025" / "01" / "vacancies.html"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "seat": [1, 2],
            "court": ["9th Circuit", "DC Circuit"],
            "vacancy_date": ["2025-01-15", "2025-02-20"],
        }
    )


@pytest.fixture
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def year_archive_paths(fixtures_dir):
    """Return a list of paths to year archive HTML files."""
    pages_dir = fixtures_dir / "pages"
    return list(pages_dir.glob("archive_*.html"))


# Tests

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
    mock_get.assert_called_once_with("http://example.com", timeout=30)


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


def test_extract_vacancy_table(real_vacancy_html):
    """Test extraction of vacancy data from HTML using real fixture."""
    records = extract_vacancy_table(real_vacancy_html)
    assert len(records) > 0

    # Check that we have expected fields based on the actual HTML structure
    expected_fields = {
        "court",          # Court name/district
        "vacancy_date",   # Date the vacancy occurred
        "incumbent",      # Name of the outgoing judge
        "nominee",        # Name of the nominee (if any)
        "vacancy_reason", # Reason for the vacancy
        "nomination_date",# Date of nomination (if any)
        "confirmation_date" # Date of confirmation (if any)
    }
    
    for record in records:
        # At least one of these fields should be present in each record
        assert any(field in record for field in expected_fields), \
            f"No expected fields found in record: {record}"
            
        # Check that date fields can be parsed if they exist
        for date_field in ["vacancy_date", "nomination_date", "confirmation_date"]:
            if date_field in record and record[date_field]:
                try:
                    datetime.strptime(record[date_field], "%m/%d/%Y")
                except ValueError:
                    assert False, f"Invalid date format in {date_field}: {record[date_field]}"


def test_records_to_dataframe(sample_dataframe):
    """Test conversion of records to DataFrame."""
    records = [
        {"seat": "1", "court": "9th Circuit", "vacancy_date": "2025-01-15"},
        {"seat": "2", "court": "DC Circuit", "vacancy_date": "2025-02-20"},
    ]
    df = records_to_dataframe(records)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["seat", "court", "vacancy_date"]
    assert df["seat"].dtype == "int64"


def test_save_to_csv(tmp_path, sample_dataframe):
    """Test saving DataFrame to CSV."""
    test_file = tmp_path / "test_output.csv"

    # Test successful save
    save_to_csv(sample_dataframe, test_file)
    assert test_file.exists()

    # Verify content
    df = pd.read_csv(test_file)
    assert len(df) == 2
    assert list(df.columns) == ["seat", "court", "vacancy_date"]


@patch("pandas.DataFrame.to_csv")
def test_save_to_csv_error(mock_to_csv):
    """Test error handling when saving DataFrame."""
    # Mock the to_csv method to raise an exception
    mock_to_csv.side_effect = Exception("Test error")

    # Test that the exception is properly caught and re-raised
    with pytest.raises(DataPipelineError) as exception_info:
        save_to_csv(pd.DataFrame(), "/invalid/path/test.csv")
    assert "Failed to save CSV" in str(exception_info.value)


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
    
    # TODO: ensure that this test case or a separate test case covers that the unit under test generates and/or fetches URLs for 2009 July and newer, i.e. months 07 and newer, excluding months 01 through 06.



