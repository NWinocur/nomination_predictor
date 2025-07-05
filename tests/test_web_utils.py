"""Tests for the web_utils module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from nomination_predictor.web_utils import (
    FetchError,
    ParseError,
    extract_month_links,
    fetch_html,
    generate_or_fetch_archive_urls,
    validate_url,
)


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
    mock_get.side_effect = requests.Timeout("Request timed out")
    
    with pytest.raises(FetchError):
        fetch_html("http://example.com", max_retries=2)
    
    assert mock_get.call_count == 2  # Initial + 1 retry


def test_generate_or_fetch_archive_urls():
    """Test generation of archive URLs."""
    urls = generate_or_fetch_archive_urls()
    assert len(urls) > 0
    assert all(url.startswith("https://www.uscourts.gov") for url in urls)
    assert all("archive-judicial-vacancies?year=" in url for url in urls)


def test_generate_or_fetch_archive_urls_includes_current_year():
    """Test that the current year is included in the generated URLs."""
    current_year = 2025  # Assuming current year is 2025 as per the test data
    urls = generate_or_fetch_archive_urls()
    assert any(f"year={current_year}" in url for url in urls)


# Fixture for year archive paths
@pytest.fixture
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def year_archive_paths(fixtures_dir):
    """Return a list of paths to year archive HTML files."""
    pages_dir = fixtures_dir / "pages"
    return list(pages_dir.glob("archive_*.html"))


def test_extract_month_links_with_real_fixtures(year_archive_paths):
    """Test extraction of month links from real year archive pages."""
    # Skip if no fixture files found
    if not year_archive_paths:
        pytest.skip("No year archive fixtures found. See readme for how to prep test fixtures.")

    for path in year_archive_paths:
        html = path.read_text(encoding="utf-8")
        
        # Extract year from filename (e.g., archive_2024.html -> 2024)
        year = int(path.stem.split('_')[-1])
        
        try:
            month_links = extract_month_links(html)
            
            # Basic validation of the results
            assert isinstance(month_links, list)
            
            # For each month link, verify structure and content
            for link in month_links:
                assert "url" in link, f"Missing 'url' in link: {link}"
                assert "month" in link, f"Missing 'month' in link: {link}"
                assert "year" in link and link["year"] is None, f"Year should be None in link: {link}"
                
                # Verify month name is valid
                assert link["month"].lower() in [
                    'january', 'february', 'march', 'april', 'may', 'june',
                    'july', 'august', 'september', 'october', 'november', 'december'
                ], f"Invalid month name: {link['month']}"
                
                # Verify URL structure
                assert link["url"].startswith(("/", "http://", "https://")), \
                    f"URL should start with /, http://, or https://: {link['url']}"
                
                # Verify the URL contains the year
                assert str(year) in link["url"], f"Year {year} not in URL: {link['url']}"
                
                # Verify the URL contains a month number (1-12)
                month_num = [
                    'january', 'february', 'march', 'april', 'may', 'june',
                    'july', 'august', 'september', 'october', 'november', 'december'
                ].index(link["month"].lower()) + 1
                
                # Split the URL path and get the segment containing the month number
                # (second-to-last segment before the filename/query)
                url_parts = link["url"].rstrip('/').split('/')
                month_segment = url_parts[-2] if len(url_parts) > 1 else url_parts[-1]
                
                assert str(month_num) in month_segment, \
                    f"Month number {month_num} not found in URL segment: {month_segment} (URL: {link['url']})"
                
        except Exception as e:
            pytest.fail(f"Failed to process {path.name}: {str(e)}")


def test_extract_month_links_invalid_html():
    """Test extraction with invalid HTML."""
    with pytest.raises(ParseError):
        extract_month_links("<html><body>No table here</body></html>")


def test_extract_month_links_missing_table():
    """Test extraction when the expected table is missing."""
    html = """
    <html>
        <body>
            <div class="layout-content">
                <p>No table here</p>
            </div>
        </body>
    </html>
    """
    with pytest.raises(ParseError, match="Could not find the months table"):
        extract_month_links(html)
