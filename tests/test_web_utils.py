"""Tests for the web_utils module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from nomination_predictor.web_utils import (
    FetchError,
    fetch_html,
    generate_month_links,
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
    mock_get.assert_called_once_with("http://example.com", timeout=60)


@patch("requests.get")
def test_fetch_html_retry_on_timeout(mock_get):
    """Test fetch_html retries on timeout."""
    mock_get.side_effect = requests.Timeout("Request timed out")
    
    with pytest.raises(FetchError):
        fetch_html("http://example.com", max_retries=2)
    
    assert mock_get.call_count == 3


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


def test_generate_month_links_structure():
    """Test the structure of generated month links."""
    # Test with a specific year
    year = 2025
    month_links = generate_month_links(year)
    
    # Should return exactly 12 months
    assert len(month_links) == 12, "Should generate links for all 12 months"
    
    # Should have all required fields in each link
    for link in month_links:
        assert 'url' in link, "Link should have 'url' field"
        assert 'month' in link, "Link should have 'month' field"
        assert 'year' in link, "Link should have 'year' field"


def test_generate_month_links_month_format():
    """Test the format of month values in generated links."""
    year = 2025
    month_links = generate_month_links(year)
    
    # Check month format (01-12) and uniqueness
    months = set()
    for link in month_links:
        month = link['month']
        assert month in [f"{i:02d}" for i in range(1, 13)], f"Invalid month format: {month}"
        months.add(month)
    
    # Should have all 12 unique months
    assert len(months) == 12, "Should have all 12 unique months"


def test_generate_month_links_year_handling():
    """Test that year is correctly handled in different formats."""
    test_cases = [
        2020,  # Recent past
        2025,  # Current/future
        2009,  # further past
    ]
    
    for year in test_cases:
        month_links = generate_month_links(year)
        
        # All years should match input year (as string)
        assert all(link['year'] == str(year) for link in month_links), \
            f"All years should match input year {year}"
        
        # URL should contain the correct year and month
        first_link = month_links[0]
        assert f"/{year}/" in first_link['url'], \
            f"URL should contain the year {year}"
        assert f"{first_link['month']}/vacancies" in first_link['url'], \
            "URL should contain month and 'vacancies'"


@pytest.mark.parametrize("page_type,expected_suffix", [
    ("vacancies", "vacancies"),
    ("confirmations", "confirmations"),
    ("emergencies", "emergencies"),
])
def test_generate_month_links_url_structure(page_type, expected_suffix):
    """Test the structure of generated URLs for different page types."""
    year = 2025
    month_links = generate_month_links(year, page_type=page_type)
    
    base_url = "/judges-judgeships/judicial-vacancies/archive-judicial-vacancies"
    
    for link in month_links:
        # URL should start with the base path
        assert link['url'].startswith(f"{base_url}/"), \
            f"URL should start with {base_url}/"
            
        # URL should contain year, month, and the correct page type
        expected_url_part = f"{year}/{link['month']}/{expected_suffix}"
        assert expected_url_part in link['url'], \
            f"URL should contain {expected_url_part}"


def test_generate_month_links_default_page_type():
    """Test that the default page type is 'vacancies'."""
    year = 2025
    month_links = generate_month_links(year)
    
    # Default should be 'vacancies'
    for link in month_links:
        assert "/vacancies" in link['url'], \
            "Default page type should be 'vacancies'"


def test_generate_month_links_invalid_page_type():
    """Test that an invalid page type raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid page_type"):
        generate_month_links(2025, page_type="invalid_type")


def test_generate_month_links_2009():
    """
    Test that month links for 2009 only include July (07) through December (12).
    
    Verifies that:
    1. For 2009, only months 07-12 are included
    2. For other years, all months (01-12) are included
    """
    # Test with 2009 - should only return months 07-12
    month_links_2009 = generate_month_links(2009, "vacancies")
    months_2009 = {link['month'] for link in month_links_2009}
    expected_months_2009 = {f"{m:02d}" for m in range(7, 13)}  # 07-12
    assert months_2009 == expected_months_2009, \
        f"For 2009, expected months {expected_months_2009}, got {months_2009}"
    
    # Test with 2010 - should return all months 01-12
    month_links_2010 = generate_month_links(2010, "vacancies")
    months_2010 = {link['month'] for link in month_links_2010}
    expected_months_2010 = {f"{m:02d}" for m in range(1, 13)}  # 01-12
    assert months_2010 == expected_months_2010, \
        f"For 2010, expected months {expected_months_2010}, got {months_2010}"
    
    # Verify the URL format for a sample month
    sample_month = month_links_2009[0]
    assert sample_month['url'] == (
        "/judges-judgeships/judicial-vacancies/archive-judicial-vacancies/2009/07/vacancies"
    )
    assert sample_month['month'] == '07'
    assert sample_month['year'] == '2009'
    assert sample_month['page_type'] == 'vacancies'
