"""Tests for the emergencies_scraper module."""

from pathlib import Path
from typing import Any, Dict

from bs4 import BeautifulSoup
import pytest

from nomination_predictor.emergencies_scraper import (
    _detect_table_format,
    extract_emergencies_table,
    is_valid_court_identifier,
)


def get_pre_downloaded_emergencies_html_from(year: int, month_num: str) -> str:
    """Return the content of a real emergencies page from the fixtures."""
    path = Path(__file__).parent / "fixtures" / "pages" / str(year) / month_num / "emergencies.html"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def validate_emergency_record(record: Dict[str, Any]) -> None:
    """Validate the structure and content of an emergency record.
    
    Args:
        record: The record to validate
        
    Raises:
        AssertionError: If the record is invalid
    """
    assert isinstance(record, dict), "Record should be a dictionary"
    
    # Required fields - based on actual emergencies.html structure
    required_fields = [
        'circuit_district',
        'vacancy_created_by_judge_name',
        'reason',
        'vacancy_date',
        'days_pending',
        #'weighted_filings_per_judgeship', # optional
        #'adjusted_filings_per_panel' # optional
    ]
    
    # Check required fields exist
    for field in required_fields:
        assert field in record, f"Missing required field: {field}"
        assert record[field] is not None, f"Required field '{field}' should not be None"
    
    # Check field types for required fields
    field_specs = {
        'circuit_district': (str,),
        'vacancy_created_by_judge_name': (str,),
        'reason': (str,),
        'vacancy_date': (str,),
        'days_pending': (str, int)  # Can be either string or int
    }
    
    for field, valid_types in field_specs.items():
        value = record[field]
        if not isinstance(value, valid_types):
            type_names = [t.__name__ for t in valid_types]
            raise AssertionError(
                f"Field '{field}' should be {' or '.join(type_names)}, "
                f"got {type(value).__name__} with value: {value!r}"
            )
    
    # Optional fields (may be missing or empty)
    optional_fields = {
        'weighted_filings_per_judgeship': str,
        'adjusted_filings_per_panel': str,
    }
    
    # Check optional fields if they exist
    for field, field_type in optional_fields.items():
        if field in record and record[field] is not None:
            assert isinstance(record[field], field_type), \
                f"Optional field '{field}' should be {field_type.__name__} if present"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


# Test cases for emergency table extraction
# Format: (year, month, expected_emergencies)
JUDICIAL_EMERGENCY_QUANTITIES = [
    (2010, "01", 33),
    (2015, "01", 12),
    (2015, "03", 21),
    (2015, "06", 24),
    (2024, "09", 22),
    (2024, "06", 24),
]


@pytest.mark.parametrize("year,month,expected_emergencies", JUDICIAL_EMERGENCY_QUANTITIES)
def test_extract_emergencies_table(year: int, month: str, expected_emergencies: int) -> None:
    """Test extraction of emergency declaration data from HTML using real fixtures."""
    # Get the HTML content from fixtures
    html = get_pre_downloaded_emergencies_html_from(year, month)
    
    # Extract the data
    records = extract_emergencies_table(html)
    
    # Basic validation
    assert isinstance(records, list), "Should return a list of records"
    assert len(records) == expected_emergencies, \
        f"Expected {expected_emergencies} records, got {len(records)}"
    
    # Validate each record
    for record in records:
        validate_emergency_record(record)
        
    # Additional validation specific to the test case
    if year == 2010 and month == "01":
        # Check some known values from the 2010-01 data
        ca9_record = next((r for r in records if r['circuit_district'] == '09 - CA'), None)
        assert ca9_record is not None, "Expected to find CA-9 record"
        assert ca9_record['vacancy_created_by_judge_name'], "Judge name should not be empty"
        assert int(ca9_record['days_pending']) > 0, "Days pending should be positive"


def test_detect_table_format_modern():
    """Test that modern tables with thead/tbody (e.g. 2015 June and onwards) are correctly identified."""
    html = """
    <table>
        <thead>
            <tr><th>Circuit/District</th><th>Title</th></tr>
        </thead>
        <tbody>
            <tr><td>01 - CCA</td><td>Circuit Judge</td></tr>
        </tbody>
    </table>
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    assert _detect_table_format(table) == 'modern'


def test_detect_table_format_legacy():
    """Test that legacy tables without thead/tbody (e.g. 2015 May and prior) are correctly identified."""
    html = """
    <table>
        <tr><th>Circuit/District</th><th>Title</th></tr>
        <tr><td>01 - CCA</td><td>Circuit Judge</td></tr>
    </table>
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    assert _detect_table_format(table) == 'legacy'


def test_detect_table_format_empty():
    """Test that empty tables default to modern format."""
    html = "<table></table>"
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    assert _detect_table_format(table) == 'modern'


def test_is_valid_court_identifier():
    """Test validation of court identifiers."""
    assert is_valid_court_identifier("01 - CCA") is True
    assert is_valid_court_identifier("09 - CA") is True
    assert is_valid_court_identifier("DC - DC") is True
    assert is_valid_court_identifier("IT") is True  # International Trade
    assert is_valid_court_identifier("CL") is True  # Federal Claims
    assert is_valid_court_identifier("SC") is True  # Supreme Court
    assert is_valid_court_identifier("Invalid") is False
    assert is_valid_court_identifier("") is False
    assert is_valid_court_identifier(None) is False
