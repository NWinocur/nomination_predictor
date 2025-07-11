"""Tests for the vacancy_scraper module."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest


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