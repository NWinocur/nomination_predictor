"""Tests for the vacancy_scraper module."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from bs4 import BeautifulSoup
import pytest

from nomination_predictor.vacancy_scraper import _detect_table_format, extract_vacancy_table


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
    
    # Optional fields (may be missing or empty)
    optional_fields = {
        'nominee': str,
        'nomination_date': str,
        'confirmation_date': str,
        'incumbent': str  # Optional field
    }
    
    # Check field types for required fields
    for field, field_type in {
        'court': str,
        'vacancy_date': str,
        'vacancy_reason': str,
    }.items():
        assert isinstance(record[field], field_type), \
            f"Field '{field}' has incorrect type. Expected {field_type}, got {type(record[field])}"
    
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


@pytest.fixture
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("year,month,expected_vacancies,expected_nominees_pending", [
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
    
    # 2021-2025: Recent years with potential format changes
    (2021, "01", 49, 26),    # Recent year with typical data
    (2022, "01", 76, 25),    # Recent year with typical data
    (2024, "01", 61, 25),    # Recent year with typical data
    (2025, "01", 40, 0),     # Most recent available data
])
def test_extract_vacancy_table(year, month, expected_vacancies, expected_nominees_pending):
    """Test extraction of vacancy data from HTML using real fixtures from different years."""
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