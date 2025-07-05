"""Tests for the dataset module."""

from datetime import datetime
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pandas as pd
import pytest

# Import the module to test
from nomination_predictor.dataset import (
    DataPipelineError,
    extract_vacancy_table,
    records_to_dataframe,
    save_to_csv,
)


def validate_vacancy_record(record: Dict[str, str]) -> None:
    """Validate that a vacancy record has the expected structure and data types.
    
    Args:
        record: Dictionary containing vacancy data to validate
        
    Raises:
        AssertionError: If the record fails validation
    """
    assert isinstance(record, dict), "Record should be a dictionary"
    
    # Required fields
    required_fields = ['court', 'vacancy_date', 'incumbent', 'vacancy_reason']
    for field in required_fields:
        assert field in record, f"Missing required field: {field}"
    
    # Validate date format if present
    if record.get('vacancy_date'):
        try:
            datetime.strptime(record['vacancy_date'], "%m/%d/%Y")
        except ValueError:
            assert False, f"Invalid date format in vacancy_date: {record['vacancy_date']}"
    
    # Optional fields should have appropriate types if present
    optional_fields = {
        'nominee': str,
        'nomination_date': str,
        'confirmation_date': str
    }
    
    for field, field_type in optional_fields.items():
        if field in record and record[field] is not None:
            assert isinstance(record[field], field_type), f"{field} should be {field_type.__name__}"


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
@pytest.mark.parametrize("year,month,expected_vacancies,expected_nominees_pending", [
    # Test different HTML formats across years
    (2010, "01", 110, 53),    # Older format
    (2014, "01", 92, 53),    # Older format
    (2015, "01", 43, 0),    # Transitional period
    (2016, "01", 75, 34),    # Modern format
    (2020, "01", 80, 51),    # Modern format (mid-range)
    (2025, "01", 40, 0),
])
def test_extract_vacancy_table(year, month, expected_vacancies, expected_nominees_pending):
    """Test extraction of vacancy data from HTML using real fixtures from different years.
    
    Tests multiple years to ensure compatibility with different HTML formats.
    
    Args:
        year: The year to test
        month: The month to test (as 'MM' string)
        expected_vacancies: Expected number of vacancies (including ones with or without a nominee pending)
        expected_nominees_pending: Expected number of nominees pending (vacancies with a nominee named and nomination date)
    """

    html_content = get_pre_downloaded_vacancies_html_from(year, month)
    records = extract_vacancy_table(html_content)
    
    # Check record count is within expected range
    assert expected_vacancies == len(records), f"Expected {expected_vacancies} records for {month}/{year}, got {len(records)}"
    
    # Check that we have expected fields based on the actual HTML structure
    expected_fields = {
        "court",          # Court name/district
        "vacancy_date",   # Date the vacancy occurred
        "incumbent",      # Name of the outgoing judge
        "vacancy_reason", # Reason for the vacancy
    }
    
    for record in records:
        # Validate the record structure and content
        validate_vacancy_record(record)
        
        # Check that all required fields are present and non-empty
        for field in expected_fields:
            assert field in record, f"Missing expected field '{field}' in record: {record}"
            assert record[field], f"Empty value for required field '{field}' in record: {record}"
    
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
