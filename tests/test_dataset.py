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
    FetchError,
    extract_month_links,
    extract_vacancy_table,
    fetch_html,
    generate_or_fetch_archive_urls,
    records_to_dataframe,
    save_to_csv,
    validate_url,
)


# Fixtures
@pytest.fixture
def example_page_path():
    """Return the path to the example page fixture."""
    return Path(__file__).parent / "fixtures" / "example_page.html"


@pytest.fixture
def example_page_content():
    """Return the content of the example page fixture with realistic data.
    
    This fixture provides HTML content that matches the structure of actual confirmation pages
    from the US Courts website, including all the expected fields from the data dictionary.
    """
    return """
    <html>
      <head>
        <title>Judicial Vacancies - March 2025 | United States Courts</title>
      </head>
      <body>
        <div class="main-content">
          <h1>Judicial Vacancies - March 2025</h1>
          
          <div class="vacancy-summary">
            <p>Showing vacancies as of March 1, 2025</p>
          </div>
          
          <table class="responsive-table">
            <thead>
              <tr>
                <th>Court</th>
                <th>Vacancy Date</th>
                <th>Incumbent</th>
                <th>Vacancy Reason</th>
                <th>Nominee</th>
                <th>Nomination Date</th>
                <th>Confirmation Date</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>11 - FL-M</td>
                <td>01/15/2025</td>
                <td>Smith,John A.</td>
                <td>Senior Status</td>
                <td>Burdell,James P.</td>
                <td>02/10/2025</td>
                <td>03/15/2025</td>
              </tr>
              <tr>
                <td>09 - MT</td>
                <td>12/20/2024</td>
                <td>Johnson,Mary L.</td>
                <td>Retired</td>
                <td>Williams,Robert</td>
                <td>01/05/2025</td>
                <td></td>
              </tr>
            </tbody>
          </table>
        </div>
      </body>
    </html>
    """


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


@pytest.fixture
def year_2000_archive(fixtures_dir):
    """Return the content of the year 2000 archive page."""
    path = fixtures_dir / "pages" / "archive_2000.html"
    return path.read_text(encoding="utf-8")


# Tests
@pytest.mark.parametrize(
    "url,valid",
    [
        ("http://example.com", True),
        ("https://example.com", True),
        ("ftp://example.com", False),
        ("not-a-url", False),
    ],
)
def test_validate_url(url, valid):
    """Test URL validation."""
    if valid:
        assert validate_url(url) is True
    else:
        assert validate_url(url) is False


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
def test_fetch_html_http_error_4xx(mock_get):
    """Test fetch_html doesn't retry on 4xx errors (except 429)."""
    # Create a mock response with status code 404
    mock_response = MagicMock()
    mock_response.status_code = 404

    # Create an HTTPError with the response attached
    http_error = requests.HTTPError("404 Client Error")
    http_error.response = mock_response
    mock_response.raise_for_status.side_effect = http_error

    # Set up the mock to return our error response
    mock_get.return_value = mock_response

    # The function should raise FetchError without retrying
    with pytest.raises(FetchError, match="Failed to fetch"):
        fetch_html("http://example.com/not-found", max_retries=2)

    # Should only be called once (no retries for 4xx errors)
    assert mock_get.call_count == 1


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


def test_extract_vacancy_table(example_page_content):
    """Test extraction of vacancy data from HTML."""
    records = extract_vacancy_table(example_page_content)
    assert len(records) > 0

    # Check that we have expected fields based on the actual HTML structure
    # These fields should be present in the confirmation pages
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
    with pytest.raises(DataPipelineError) as excinfo:
        save_to_csv(pd.DataFrame(), "/invalid/path/test.csv")
    assert "Failed to save CSV" in str(excinfo.value)


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
    
    # TODO: ensure that this test case or a separate test case covers that the unit under test generates and/or fetches URLs for 2009 July and newer, i.e. months 07 and newer, excluding monts 01 through 06.


def test_extract_month_links_with_real_fixtures(year_archive_paths):
    """Test extraction of month links from real year archive pages."""
    # Skip if no fixture files found
    if not year_archive_paths:
        pytest.skip("No year archive fixtures found")

    for path in year_archive_paths:
        year = int(path.stem.split("_")[1])  # Extract year from filename
        html = path.read_text(encoding="utf-8")

        # Extract month links
        try:
            month_links = extract_month_links(html)

            # Basic validation of the results
            assert isinstance(month_links, list)

            # Check each link has the expected structure
            for link in month_links:
                assert "url" in link
                assert "month" in link
                assert "year" in link and link["year"] is None

                # URL should be an absolute URL
                assert link["url"].startswith("http")

                # Month should contain a month name and possibly a year
                month_names = [
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "june",
                    "july",
                    "august",
                    "september",
                    "october",
                    "november",
                    "december",
                ]
                assert any(month in link["month"].lower() for month in month_names)

        except Exception as e:
            pytest.fail(f"Failed to process {path.name}: {str(e)}")


def test_extract_month_links_specific_year(year_2000_archive):
    """Test extraction from a specific year's archive page."""
    month_links = extract_month_links(year_2000_archive)

    # Should find month links for year 2000
    assert len(month_links) > 0

    # Check that we have links for all 12 months
    months = [link["month"].lower() for link in month_links]
    for month in [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]:
        assert any(month in m for m in months), f"Missing {month} in month links"

    # Check URLs contain the correct year
    urls = " ".join(link["url"] for link in month_links)
    assert "/2000/" in urls or "year=2000" in urls


def test_extract_month_links_empty_input():
    """Test behavior with empty HTML input."""
    assert extract_month_links("") == []


def test_extract_month_links_no_links():
    """Test behavior when no month links are present."""
    html = """
    <html>
    <body>
        <h1>No Links Here</h1>
        <p>This page doesn't contain any month links.</p>
    </body>
    </html>
    """
    assert extract_month_links(html) == []
