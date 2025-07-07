"""Tests for the dataset module."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pandas as pd
import pytest

# Import the module to test
from nomination_predictor.dataset import (
    DataPipelineError,
    records_to_dataframe,
    save_to_csv,
)


def validate_vacancy_record(record: Dict[str, Any]) -> None:
    """Validate the structure and content of a vacancy record.
    
    Args:
        record: The record to validate
        
    Raises:
        AssertionError: If the record is invalid
    """
    assert isinstance(record, dict), "Record should be a dictionary"
    
    # Define required fields and their expected types
    required_field_types = {
        'court': str,
        'vacancy_date': str, # directly from our Internet-hosted data ource this is presented to us as a string.  Conversion to datetime can happen during data cleaning & transformations.
        'vacancy_reason': str,
    }
    
    # Check required fields exist and have correct types
    for field, field_type in required_field_types.items():
        assert field in record, f"Missing required field: {field}"
        assert isinstance(record[field], field_type), \
            f"Field '{field}' has incorrect type. Expected {field_type}, got {type(record[field])}"
    
    # Optional fields (may be missing or empty)
    optional_fields = {
        'nominee': str,
        'nomination_date': str,
        'confirmation_date': str,
        'incumbent': str  # Now optional
    }
    
    for field, field_type in optional_fields.items():
        if field in record and record[field]:  # Only validate if field exists and is non-empty
            assert isinstance(record[field], field_type), \
                f"Field '{field}' has incorrect type. Expected {field_type}, got {type(record[field])}"
    
    # Validate date formats if present
    date_fields = ['vacancy_date', 'nomination_date', 'confirmation_date']
    for date_field in date_fields:
        if date_field in record and record[date_field]:
            try:
                datetime.strptime(record[date_field], '%m/%d/%Y')
            except ValueError:
                assert False, f"Invalid date format in field '{date_field}': {record[date_field]}. Expected MM/DD/YYYY"


def get_pre_downloaded_vacancies_html_from(year, month_num):
    """Return the content of a real vacancies page from the fixtures."""
    path = Path(__file__).parent / "fixtures" / "pages" / str(year) / month_num / "vacancies.html"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# Fixtures
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
    """Test saving DataFrame to CSV with pipe delimiter."""
    test_file = tmp_path / "test_output.csv"

    # Test successful save
    save_to_csv(sample_dataframe, test_file)
    assert test_file.exists()

    # Verify content with pipe delimiter
    df = pd.read_csv(test_file, sep='|')
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