"""Tests for the dataset module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

# Import the module to test
from nomination_predictor.dataset import (
    DataPipelineError,
    FetchError,
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
def example_page_content(example_page_path):
    """Return the content of the example page fixture."""
    return example_page_path.read_text()

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'seat': [1, 2],
        'court': ['9th Circuit', 'DC Circuit'],
        'vacancy_date': ['2025-01-15', '2025-02-20']
    })

# Tests
@pytest.mark.parametrize("url,valid", [
    ("http://example.com", True),
    ("https://example.com", True),
    ("ftp://example.com", False),
    ("not-a-url", False),
])
def test_validate_url(url, valid):
    """Test URL validation."""
    if valid:
        assert validate_url(url) is True
    else:
        assert validate_url(url) is False

@patch('requests.get')
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

@patch('requests.get')
def test_fetch_html_retry_on_timeout(mock_get):
    """Test fetch_html retries on timeout."""
    # Mock responses: timeout once, then succeed
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html>Success</html>"
    
    mock_get.side_effect = [
        requests.exceptions.Timeout("Connection timed out"),
        mock_response
    ]
    
    result = fetch_html("http://example.com", max_retries=2, retry_delay=0.1)
    assert result == "<html>Success</html>"
    assert mock_get.call_count == 2

@patch('requests.get')
def test_fetch_html_retry_on_connection_error(mock_get):
    """Test fetch_html retries on connection error."""
    # Mock responses: connection error once, then succeed
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html>Success</html>"
    
    mock_get.side_effect = [
        requests.exceptions.ConnectionError("Connection failed"),
        mock_response
    ]
    
    result = fetch_html("http://example.com", max_retries=2, retry_delay=0.1)
    assert result == "<html>Success</html>"
    assert mock_get.call_count == 2

@patch('requests.get')
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

@patch('requests.get')
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

@patch('requests.get')
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
    expected_fields = {'court', 'vacancy_date', 'nominating_president', 'nominee', 'status'}
    for record in records:
        assert all(field in record for field in expected_fields), \
            f"Missing fields in record: {record}"

def test_records_to_dataframe(sample_dataframe):
    """Test conversion of records to DataFrame."""
    records = [
        {'seat': '1', 'court': '9th Circuit', 'vacancy_date': '2025-01-15'},
        {'seat': '2', 'court': 'DC Circuit', 'vacancy_date': '2025-02-20'}
    ]
    df = records_to_dataframe(records)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ['seat', 'court', 'vacancy_date']
    assert df['seat'].dtype == 'int64'

def test_save_to_csv(tmp_path, sample_dataframe):
    """Test saving DataFrame to CSV."""
    test_file = tmp_path / "test_output.csv"
    
    # Test successful save
    save_to_csv(sample_dataframe, test_file)
    assert test_file.exists()
    
    # Verify content
    df = pd.read_csv(test_file)
    assert len(df) == 2
    assert list(df.columns) == ['seat', 'court', 'vacancy_date']

@patch('pandas.DataFrame.to_csv')
def test_save_to_csv_error(mock_to_csv):
    """Test error handling when saving DataFrame."""
    # Mock the to_csv method to raise an exception
    mock_to_csv.side_effect = Exception("Test error")
    
    # Test that the exception is properly caught and re-raised
    with pytest.raises(DataPipelineError) as excinfo:
        save_to_csv(pd.DataFrame(), "/invalid/path/test.csv")
    assert "Failed to save CSV" in str(excinfo.value)

def test_generate_or_fetch_archive_urls():
    """Test URL generation for archive pages."""
    # This is a simple test - in practice, you might want to mock the web request
    # or use a fixture with a saved response
    urls = generate_or_fetch_archive_urls()
    assert isinstance(urls, list)
    assert all(isinstance(url, str) for url in urls)
    assert all(url.startswith('http') for url in urls)
