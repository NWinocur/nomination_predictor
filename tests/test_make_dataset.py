"""Tests for the make_dataset module."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import re
import os
import time
import pandas as pd
import pytest
import requests


# Import the module to test
from data import make_dataset


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

@pytest.mark.parametrize("year", [2025, 2020, 2015, 2010])  # Test multiple years
@patch('data.make_dataset.requests.get')
def test_generate_or_fetch_archive_urls_years(mock_get, year):
    """Test URL generation for different years."""
    # Setup mock response with dynamic year
    mock_response = MagicMock()
    mock_response.text = f"""
    <html>
      <body>
        <a href="/{year}/vacancies">Vacancies</a>
        <a href="/{year}/emergencies">Emergencies</a>
        <a href="/{year}/confirmations">Confirmations</a>
        <a href="/{year}/other">Should be ignored</a>
      </body>
    </html>
    """
    mock_get.return_value = mock_response
    
    # Test
    urls = make_dataset.generate_or_fetch_archive_urls()
    
    # Verify
    assert len(urls) == 3
    assert all(str(year) in url for url in urls)
    assert any("vacancies" in url for url in urls)
    assert any("emergencies" in url for url in urls)
    assert any("confirmations" in url for url in urls)

@patch('data.make_dataset.requests.get')
def test_generate_or_fetch_archive_urls_no_matches(mock_get):
    """Test when no matching URLs are found."""
    mock_response = MagicMock()
    mock_response.text = """
    <html>
      <body>
        <a href="/2025/other">Should be ignored</a>
      </body>
    </html>
    """
    mock_get.return_value = mock_response
    
    with pytest.raises(make_dataset.DataPipelineError, match="Unexpected error while processing archive page: No valid URLs found on the archive page"):
        make_dataset.generate_or_fetch_archive_urls()

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

def test_extract_vacancy_table(example_page_content):
    """Test table extraction from HTML with comprehensive assertions."""
    # Act
    records = make_dataset.extract_vacancy_table(example_page_content)
    
    # Assert basic structure
    assert isinstance(records, list), "Should return a list of records"
    assert len(records) == 2, "Should extract exactly 2 records"
    
    # Define expected data in a more maintainable way
    expected_records = [
        {
            "court": "9th Circuit",
            "vacancy_date": "01/15/2025",
            "nominating_president": "Biden",
            "nominee": "John Smith",
            "status": "Pending Hearing"
        },
        {
            "court": "DC Circuit",
            "vacancy_date": "02/20/2025",
            "nominating_president": "Biden",
            "nominee": "Jane Doe",
            "status": "Pending Committee Vote"
        }
    ]
    
    # Verify each record
    for expected in expected_records:
        # Find matching record by court (unique identifier)
        actual = next((r for r in records if r["court"] == expected["court"]), None)
        assert actual is not None, f"Expected record for {expected['court']} not found"
        
        # Verify all fields match
        for field, expected_value in expected.items():
            assert field in actual, f"Field '{field}' missing from record"
            assert actual[field] == expected_value, f"Mismatch in {field} for {expected['court']}"
    
    # Verify no extra fields in actual records
    for record in records:
        expected_fields = set(expected_records[0].keys())
        actual_fields = set(record.keys())
        assert actual_fields == expected_fields, f"Unexpected fields in record for {record['court']}"

def test_extract_vacancy_table_complete_data(example_page_content):
    """Test extraction with all possible fields and their properties."""
    # Act
    records = make_dataset.extract_vacancy_table(example_page_content)
    
    # Assert basic structure
    assert isinstance(records, list), "Should return a list of records"
    assert len(records) > 0, "Should return at least one record"
    
    # Define expected fields and their validation functions
    field_validations = {
        'court': {
            'required': True,
            'type': str,
            'validator': lambda x: len(x.strip()) > 0,
            'error': 'Court name should be a non-empty string'
        },
        'vacancy_date': {
            'required': True,
            'type': str,
            'validator': lambda x: bool(re.match(r'\d{1,2}/\d{1,2}/\d{4}', x)),
            'error': 'Vacancy date should be in MM/DD/YYYY format'
        },
        'nominating_president': {
            'required': True,
            'type': str,
            'validator': lambda x: len(x.strip()) > 0,
            'error': 'Nominating president should be a non-empty string'
        },
        'nominee': {
            'required': True,
            'type': str,
            'validator': lambda x: len(x.strip()) > 0,
            'error': 'Nominee name should be a non-empty string'
        },
        'status': {
            'required': True,
            'type': str,
            'validator': lambda x: len(x.strip()) > 0,
            'error': 'Status should be a non-empty string'
        }
    }
    
    # Check each record
    for i, record in enumerate(records, 1):
        # Check that all expected fields are present
        record_fields = set(record.keys())
        expected_fields = set(field_validations.keys())
        assert record_fields == expected_fields, \
            f"Record {i} has unexpected fields. " \
            f"Expected: {sorted(expected_fields)}, Got: {sorted(record_fields)}"
        
        # Validate each field
        for field, validation in field_validations.items():
            value = record[field]
            
            # Check field exists (should be true from above, but being defensive)
            assert field in record, f"Field '{field}' missing from record {i}"
            
            # Check type
            assert isinstance(value, validation['type']), \
                f"Field '{field}' should be {validation['type'].__name__}, got {type(value).__name__}"
            
            # Skip validation for None if field is not required
            if value is None and not validation['required']:
                continue
                
            # Run field-specific validation
            assert validation['validator'](value), f"{validation['error']} in record {i}: {value}"

def test_records_to_dataframe():
    """Test DataFrame conversion with comprehensive validation."""
    # Arrange
    test_records = [
        {
            "Seat": "1",
            "Court": "9th Circuit",
            "Vacancy Date": "2025-01-15",
            "Nominating President": "Biden",
            "Nominee": "John Smith",
            "Status": "Pending"
        },
        {
            "Seat": "2",
            "Court": "DC Circuit",
            "Vacancy Date": "2025-02-20",
            "Nominating President": "Biden",
            "Nominee": "Jane Doe",
            "Status": "Pending"
        }
    ]
    
    # Act
    df = make_dataset.records_to_dataframe(test_records)
    
    # Assert
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame), "Should return a pandas DataFrame"
    assert len(df) == 2, "Should have 2 rows of data"
    
    # Check column names and types
    expected_columns = {
        'Seat': 'int64',
        'Court': 'object',
        'Vacancy Date': 'object',
        'Nominating President': 'object',
        'Nominee': 'object',
        'Status': 'object'
    }
    
    # Verify all expected columns exist
    assert set(df.columns) == set(expected_columns.keys()), \
        f"Unexpected columns. Expected: {set(expected_columns.keys())}, Got: {set(df.columns)}"
    
    # Verify column data types
    for col, expected_dtype in expected_columns.items():
        assert str(df[col].dtype) == expected_dtype, \
            f"Column '{col}' should be {expected_dtype}, got {df[col].dtype}"
    
    # Verify data integrity
    for i, record in enumerate(test_records):
        for col, expected_value in record.items():
            if col == 'Seat':
                # Seat should be converted to int
                assert df.iloc[i][col] == int(expected_value), \
                    f"Seat value mismatch in row {i}"
            else:
                assert df.iloc[i][col] == expected_value, \
                    f"Value mismatch for {col} in row {i}"

def test_records_to_dataframe_empty():
    """Test DataFrame conversion with empty records."""
    # Act
    df = make_dataset.records_to_dataframe([])
    
    # Assert
    assert isinstance(df, pd.DataFrame), "Should return a pandas DataFrame"
    assert df.empty, "DataFrame should be empty"
    assert len(df.columns) == 0, "Empty DataFrame should have no columns"
    assert df.index.empty, "Empty DataFrame should have no index"
    assert df.shape == (0, 0), "Empty DataFrame should have shape (0, 0)"
    assert df.dtypes.empty, "Empty DataFrame should have no dtypes"

def test_save_to_csv(tmp_path):
    """Test saving DataFrame to CSV with comprehensive validation."""
    # Arrange
    test_data = [
        {
            "Seat": 1,
            "Court": "9th Circuit",
            "Vacancy Date": "2025-01-15",
            "Nominating President": "Biden",
            "Nominee": "John Smith",
            "Status": "Pending"
        },
        {
            "Seat": 2,
            "Court": "DC Circuit",
            "Vacancy Date": "2025-02-20",
            "Nominating President": "Biden",
            "Nominee": "Jane Doe",
            "Status": "Pending"
        }
    ]
    test_df = pd.DataFrame(test_data)
    output_file = tmp_path / "test_output" / "judicial_vacancies.csv"
    
    # Ensure the output directory doesn't exist yet
    assert not output_file.parent.exists(), "Test directory should not exist before test"
    
    # Act
    make_dataset.save_to_csv(test_df, output_file)
    
    # Assert
    # Check file was created
    assert output_file.exists(), "Output file was not created"
    assert output_file.stat().st_size > 0, "Output file is empty"
    
    # Verify file content
    df_read = pd.read_csv(output_file)
    
    # Check basic DataFrame structure
    assert isinstance(df_read, pd.DataFrame), "Should read back a pandas DataFrame"
    assert len(df_read) == len(test_df), "Number of rows should match"
    assert set(df_read.columns) == set(test_df.columns), "Column names should match"
    
    # Verify data integrity
    for col in test_df.columns:
        if col == 'Seat':
            # Check numeric columns with potential type conversion
            pd.testing.assert_series_equal(
                df_read[col], 
                test_df[col].astype(df_read[col].dtype),
                check_names=False,
                check_dtype=False
            )
        else:
            # Check string/object columns
            pd.testing.assert_series_equal(
                df_read[col], 
                test_df[col],
                check_names=False
            )
    
    # Verify file permissions (readable and writable by current user)
    assert os.access(output_file, os.R_OK), "Output file should be readable"
    assert os.access(output_file, os.W_OK), "Output file should be writable"
    
    # Test that existing file is overwritten
    original_mtime = output_file.stat().st_mtime
    time.sleep(1)  # Ensure modification time changes
    make_dataset.save_to_csv(test_df, output_file)
    assert output_file.stat().st_mtime > original_mtime, "File should be overwritten"

def test_save_to_csv_error(tmp_path):
    """Test error handling when saving CSV with various error conditions."""
    # Arrange
    test_df = pd.DataFrame({
        'Seat': [1, 2],
        'Court': ['9th Circuit', 'DC Circuit'],
        'Status': ['Pending', 'Pending']
    })
    
    # Test directory creation permission error
    with patch('data.make_dataset.Path.mkdir') as mock_mkdir:
        mock_mkdir.side_effect = PermissionError("Permission denied")
        with pytest.raises(IOError, match="Failed to create directory"):
            make_dataset.save_to_csv(test_df, "/invalid/path/output.csv")
    
    # Test file write permission error
    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        mock_to_csv.side_effect = PermissionError("Permission denied")
        with pytest.raises(IOError, match="Error saving DataFrame to"):
            output_file = tmp_path / "output.csv"
            make_dataset.save_to_csv(test_df, output_file)
    
    # Test invalid DataFrame input
    with pytest.raises(ValueError, match="Expected pandas DataFrame"):
        make_dataset.save_to_csv("not a dataframe", tmp_path / "output.csv")
    
    # Test invalid path type
    with pytest.raises(TypeError, match="path must be a string or Path"):
        make_dataset.save_to_csv(test_df, 12345)  # Invalid path type

@patch('data.make_dataset.generate_or_fetch_archive_urls')
@patch('data.make_dataset.fetch_html')
@patch('data.make_dataset.extract_vacancy_table')
@patch('data.make_dataset.records_to_dataframe')
@patch('data.make_dataset.save_to_csv')
def test_main(mock_save, mock_df, mock_extract, mock_fetch, mock_urls, tmp_path, example_page_content):
    """Test main pipeline execution."""
    # Setup mocks
    mock_urls.return_value = ["http://example.com/2025/vacancies"]
    mock_fetch.return_value = example_page_content
    mock_extract.return_value = [
        {
            "court": "9th Circuit", 
            "vacancy_date": "01/15/2025", 
            "nominating_president": "Biden", 
            "nominee": "John Smith", 
            "status": "Pending Hearing"
        },
        {
            "court": "DC Circuit", 
            "vacancy_date": "02/20/2025", 
            "nominating_president": "Biden", 
            "nominee": "Jane Doe", 
            "status": "Pending Committee Vote"
        }
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
    mock_extract.assert_called_once_with(example_page_content)
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
