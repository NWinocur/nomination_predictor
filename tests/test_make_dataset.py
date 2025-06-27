"""Tests for the make_dataset module."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

# Import the module to test
from data import make_dataset

# Test data
SAMPLE_HTML = """
<html>
  <body>
    <table>
      <thead>
        <tr><th>Seat</th><th>Court</th><th>Vacancy Date</th></tr>
      </thead>
      <tbody>
        <tr><td>1</td><td>9th Circuit</td><td>2025-01-15</td></tr>
        <tr><td>2</td><td>DC Circuit</td><td>2025-02-20</td></tr>
      </tbody>
    </table>
  </body>
</html>
"""

# Fixtures
@pytest.fixture
def sample_html_file(tmp_path):
    """Create a temporary HTML file with sample data."""
    file_path = tmp_path / "test.html"
    file_path.write_text(SAMPLE_HTML)
    return file_path

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'Seat': [1, 2],
        'Court': ['9th Circuit', 'DC Circuit'],
        'Vacancy Date': ['2025-01-15', '2025-02-20']
    })

# Tests
def test_validate_url():
    """Test URL validation."""
    assert make_dataset.validate_url("http://example.com") is True
    assert make_dataset.validate_url("https://example.com") is True
    assert make_dataset.validate_url("ftp://example.com") is False
    assert make_dataset.validate_url("not-a-url") is False

@patch('data.make_dataset.requests.get')
def test_generate_or_fetch_archive_urls(mock_get, tmp_path):
    """Test URL generation from archive page."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.text = """
    <html>
      <body>
        <a href="/2025/vacancies">Vacancies</a>
        <a href="/2025/emergencies">Emergencies</a>
        <a href="/2025/confirmations">Confirmations</a>
      </body>
    </html>
    """
    mock_get.return_value = mock_response
    
    # Test
    urls = make_dataset.generate_or_fetch_archive_urls()
    
    # Verify
    assert len(urls) == 3
    assert all("2025" in url for url in urls)
    assert any("vacancies" in url for url in urls)
    assert any("emergencies" in url for url in urls)
    assert any("confirmations" in url for url in urls)

@patch('data.make_dataset.requests.get')
def test_fetch_html_success(mock_get):
    """Test successful HTML fetching."""
    mock_response = MagicMock()
    mock_response.text = "<html><body>Test</body></html>"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    html = make_dataset.fetch_html("http://example.com")
    assert "<body>" in html

@patch('data.make_dataset.requests.get')
def test_fetch_html_retry(mock_get):
    """Test HTML fetching with retries."""
    # First two attempts fail, third succeeds
    mock_response_success = MagicMock()
    mock_response_success.text = "<html><body>Success</body></html>"
    mock_response_success.raise_for_status.return_value = None
    
    mock_get.side_effect = [
        requests.RequestException("First failure"),
        requests.RequestException("Second failure"),
        mock_response_success
    ]
    
    html = make_dataset.fetch_html("http://example.com")
    assert "Success" in html
    assert mock_get.call_count == 3

def test_extract_vacancy_table():
    """Test table extraction from HTML."""
    records = make_dataset.extract_vacancy_table(SAMPLE_HTML)
    
    assert len(records) == 2
    assert records[0]["Seat"] == "1"
    assert records[0]["Court"] == "9th Circuit"
    assert records[1]["Seat"] == "2"
    assert records[1]["Court"] == "DC Circuit"

def test_extract_vacancy_table_complete_data():
    """Test extraction with all possible fields"""
    html = """
    <table>
        <tr>
            <th>Court</th>
            <th>Vacancy Date</th>
            <th>Nominating President</th>
            <th>Nominee</th>
            <th>Status</th>
        </tr>
        <tr>
            <td>9th Circuit</td>
            <td>01/15/2024</td>
            <td>Biden</td>
            <td>John Smith</td>
            <td>Pending Hearing</td>
        </tr>
    </table>
    """
    result = make_dataset.extract_vacancy_table(html)
    assert len(result) == 1
    assert result[0]["court"] == "9th Circuit"
    assert result[0]["status"] == "Pending Hearing"

def test_records_to_dataframe(sample_dataframe):
    """Test DataFrame conversion."""
    records = [
        {"Seat": "1", "Court": "9th Circuit", "Vacancy Date": "2025-01-15"},
        {"Seat": "2", "Court": "DC Circuit", "Vacancy Date": "2025-02-20"}
    ]
    
    df = make_dataset.records_to_dataframe(records)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df.columns) == {"Seat", "Court", "Vacancy Date"}
    assert df["Seat"].dtype == "int64"

def test_records_to_dataframe_empty():
    """Test DataFrame conversion with empty records."""
    df = make_dataset.records_to_dataframe([])
    assert df.empty
    assert set(df.columns) == set()

def test_save_to_csv(sample_dataframe, tmp_path):
    """Test saving DataFrame to CSV."""
    output_file = tmp_path / "output.csv"
    
    # Test saving
    make_dataset.save_to_csv(sample_dataframe, output_file)
    
    # Verify file was created
    assert output_file.exists()
    
    # Verify content
    df_read = pd.read_csv(output_file)
    pd.testing.assert_frame_equal(df_read, sample_dataframe)

@patch('data.make_dataset.Path')
def test_save_to_csv_error(mock_path, sample_dataframe):
    """Test error handling when saving CSV."""
    # Setup mock to raise an error
    mock_path.return_value.parent.mkdir.side_effect = PermissionError("Permission denied")
    
    with pytest.raises(IOError, match="Permission denied"):
        make_dataset.save_to_csv(sample_dataframe, "/invalid/path/output.csv")

@patch('data.make_dataset.generate_or_fetch_archive_urls')
@patch('data.make_dataset.fetch_html')
@patch('data.make_dataset.extract_vacancy_table')
@patch('data.make_dataset.records_to_dataframe')
@patch('data.make_dataset.save_to_csv')
def test_main(mock_save, mock_df, mock_extract, mock_fetch, mock_urls, tmp_path):
    """Test main pipeline execution."""
    # Setup mocks
    mock_urls.return_value = ["http://example.com/2025/vacancies"]
    mock_fetch.return_value = SAMPLE_HTML
    mock_extract.return_value = [
        {"Seat": "1", "Court": "9th Circuit", "Vacancy Date": "2025-01-15"},
        {"Seat": "2", "Court": "DC Circuit", "Vacancy Date": "2025-02-20"}
    ]
    mock_df.return_value = pd.DataFrame(mock_extract.return_value)
    
    # Set output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Patch the output path
    with patch('data.make_dataset.Path') as mock_path:
        mock_path.return_value = output_dir / "judicial_vacancies.csv"
        
        # Run main
        make_dataset.main()
    
    # Verify function calls
    mock_urls.assert_called_once()
    mock_fetch.assert_called_once_with("http://example.com/2025/vacancies")
    mock_extract.assert_called_once_with(SAMPLE_HTML)
    mock_df.assert_called_once_with(mock_extract.return_value)
    mock_save.assert_called_once()

def test_error_handling():
    """Test custom error hierarchy."""
    # Test inheritance
    assert issubclass(make_dataset.FetchError, make_dataset.DataPipelineError)
    assert issubclass(make_dataset.ParseError, make_dataset.DataPipelineError)
    assert issubclass(make_dataset.ValidationError, make_dataset.DataPipelineError)
    
    # Test instantiation
    with pytest.raises(make_dataset.FetchError):
        raise make_dataset.FetchError("Test error")
