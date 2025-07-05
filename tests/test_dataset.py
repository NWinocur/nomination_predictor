"""Tests for the dataset module."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from bs4 import BeautifulSoup
import pandas as pd
import pytest

# Import the module to test
from nomination_predictor.dataset import (
    DataPipelineError,
    _detect_table_format,
    extract_vacancy_table,
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
    
    # Required fields
    required_fields = ['court', 'vacancy_date', 'vacancy_reason']
    for field in required_fields:
        assert field in record, f"Missing required field: {field}"
    
    # Check field types
    required_field_types = {
        'court': str,
        'vacancy_date': str,
        'vacancy_reason': str,
    }
    
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
@pytest.mark.parametrize("year,month,expected_vacancies,expected_nominees_pending", [
    # Test different HTML formats across years
    # 2010: Legacy format with header rows and specific structure
    (2010, "01", 101, 20),    # Contains a few entries from non-numbered circuits such as DC-DC
    
    # 2014: Transitional format with some legacy elements
    (2014, "01", 92, 53),    # Contains two "Director" seats with no prior incumbent
    
    # 2015: Transitional period with International Trade court seats
    (2015, "01", 43, 0),     # Includes three International Trade court seats (IT)
    
    # 2016-2017: Modern format with thead/tbody structure
    (2016, "01", 75, 34),    # Modern format with consistent structure
    (2017, "01", 112, 59),   # Modern format with higher vacancy count
    
    # 2018-2020: Modern format with various data patterns
    (2018, "01", 148, 51),   # Modern format with more complex data
    # (2019, "01", 114, 70),   # Modern format with typical data # skipped because actual web page summary line uses total including out-of-scope court types, totaling 144 when page's table only shows <100
    #(2020, "01", 80, 51),    # Modern format with mixed data # skipped because actual web page summary line uses total including out-of-scope court types
    
    # 2021-2025: Recent years with potential format changes
    (2021, "01", 49, 26),    # Recent year with typical data
    (2022, "01", 76, 25),    # Recent year with typical data
    # (2023, "01", 86, 37),    # Recent year with typical data # skipped because actual web page summary line says 37 but table shows 38
    (2024, "01", 61, 25),    # Recent year with typical data
    (2025, "01", 40, 0),     # Most recent available data
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
    try:
        html_content = get_pre_downloaded_vacancies_html_from(year, month)
    except FileNotFoundError:
        pytest.skip(f"No HTML fixture available for {month}/{year}")
        return

    records = extract_vacancy_table(html_content)
    
    # Check record count matches expected
    assert len(records) == expected_vacancies, \
        f"Expected {expected_vacancies} records for {month}/{year}, got {len(records)}"
    
    # Check number of records with nominees matches expected
    nominees_count = sum(1 for r in records if r.get('nominee') and r.get('nomination_date'))
    assert nominees_count == expected_nominees_pending, \
        f"Expected {expected_nominees_pending} nominees pending for {month}/{year}, got {nominees_count}"
    
    # Check that we have expected fields based on the actual HTML structure
    expected_fields = {
        "court",          # Court name/district
        "vacancy_date",   # Date the vacancy occurred
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


def test_detect_table_format_modern():
    """Test that modern tables with thead/tbody are correctly identified."""
    # Modern table with thead and tbody
    modern_html = """
    <table>
        <thead><tr><th>Court</th><th>Judge</th></tr></thead>
        <tbody><tr><td>1st Circuit</td><td>John Doe</td></tr></tbody>
    </table>
    """
    soup = BeautifulSoup(modern_html, 'html.parser')
    table = soup.find('table')
    assert _detect_table_format(table) == 'modern'


def test_detect_table_format_legacy():
    """Test that legacy tables with header rows are correctly identified."""
    # Legacy table with header row containing 'court'
    legacy_html = """
    <table>
        <tr><th>Court Name</th><th>Judge</th></tr>
        <tr><td>1st Circuit</td><td>John Doe</td></tr>
    </table>
    """
    soup = BeautifulSoup(legacy_html, 'html.parser')
    table = soup.find('table')
    assert _detect_table_format(table) == 'legacy'


def test_detect_table_format_legacy_with_strong():
    """Test that legacy tables with header rows using <strong> tags are identified."""
    # Legacy table with header row using <strong> tags
    legacy_strong_html = """
    <table>
        <tr><td><strong>Court</strong></td><td><strong>Judge</strong></td></tr>
        <tr><td>1st Circuit</td><td>John Doe</td></tr>
    </table>
    """
    soup = BeautifulSoup(legacy_strong_html, 'html.parser')
    table = soup.find('table')
    assert _detect_table_format(table) == 'legacy'


def test_detect_table_format_empty():
    """Test that empty tables default to modern format."""
    # Empty table
    empty_html = "<table></table>"
    soup = BeautifulSoup(empty_html, 'html.parser')
    table = soup.find('table')
    assert _detect_table_format(table) == 'modern'


def test_detect_table_format_with_real_fixtures(fixtures_dir):
    """Test detection with real fixture files to ensure compatibility."""
    # Test with a known legacy fixture (2010)
    legacy_html = (fixtures_dir / "pages" / "2010" / "01" / "vacancies.html").read_text()
    soup = BeautifulSoup(legacy_html, 'html.parser')
    table = soup.find('table')
    assert _detect_table_format(table) == 'legacy', "2010 fixture should be detected as legacy format"
    
    # Test with a known modern fixture (2024)
    modern_html = (fixtures_dir / "pages" / "2024" / "01" / "vacancies.html").read_text()
    soup = BeautifulSoup(modern_html, 'html.parser')
    table = soup.find('table')
    assert _detect_table_format(table) == 'modern', "2024 fixture should be detected as modern format"
