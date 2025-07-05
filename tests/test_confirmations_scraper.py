"""Tests for the confirmations_scraper module."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from bs4 import BeautifulSoup
import pytest

from nomination_predictor.confirmations_scraper import (
    _detect_table_format,
    extract_confirmations_table,
    is_valid_court_identifier,
)


def get_pre_downloaded_confirmations_html_from(year: int, month_num: str) -> str:
    """Return the content of a real confirmations page from the fixtures."""
    path = Path(__file__).parent / "fixtures" / "pages" / str(year) / month_num / "confirmations.html"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def validate_confirmation_record(record: Dict[str, Any]) -> None:
    """Validate the structure and content of a confirmation record.
    
    Args:
        record: The record to validate
        
    Raises:
        AssertionError: If the record is invalid
    """
    assert isinstance(record, dict), "Record should be a dictionary"
    
    # Required fields - based on confirmations.html structure
    required_fields = [
        'nominee',
        'nomination_date',
        'confirmation_date',
        'circuit_district'
    ]
    
    # Check required fields exist
    for field in required_fields:
        assert field in record, f"Missing required field: {field}"
        assert record[field], f"Required field '{field}' is empty"
    
    # Check field types for required fields
    for field, field_type in {
        'nominee': str,
        'nomination_date': str,
        'confirmation_date': str,
        'circuit_district': str
    }.items():
        assert isinstance(record[field], field_type), \
            f"Field '{field}' has incorrect type. Expected {field_type}, got {type(record[field])}"
    
    # Optional fields (may be missing or empty)
    optional_fields = {
        'incumbent': str,
        'vacancy_date': str,
        'vacancy_reason': str,
    }
    
    for field, field_type in optional_fields.items():
        if field in record and record[field]:  # Only validate if field exists and is non-empty
            assert isinstance(record[field], field_type), \
                f"Field '{field}' has incorrect type. Expected {field_type}, got {type(record[field])}"
    
    # Validate date formats if present
    date_fields = {
        'nomination_date': "%m/%d/%Y",
        'confirmation_date': "%m/%d/%Y",
        'vacancy_date': "%m/%d/%Y"
    }
    
    for date_field, date_format in date_fields.items():
        if date_field in record and record[date_field]:
            try:
                datetime.strptime(record[date_field].strip(), date_format)
            except ValueError:
                assert False, f"Invalid date format in field '{date_field}': {record[date_field]}. Expected {date_format}"


@pytest.fixture
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("year,month,expected_confirmations", [
    # Test different years with expected number of confirmations
    # These values will need to be updated based on actual fixture data
    (2010, "01", 10),
    (2015, "06", 4),
    (2020, "12", 149),
])
def test_extract_confirmations_table(year, month, expected_confirmations):
    """Test extraction of confirmation data from HTML using real fixtures."""
    try:
        html_content = get_pre_downloaded_confirmations_html_from(year, month)
    except FileNotFoundError:
        pytest.skip(f"No HTML fixture available for confirmations in {month}/{year}")
        return

    records = extract_confirmations_table(html_content)
    
    # Check that we got the expected number of records
    assert len(records) == expected_confirmations, \
        f"Expected {expected_confirmations} confirmations for {month}/{year}, got {len(records)}"
    
    # Validate each record
    for record in records:
        validate_confirmation_record(record)
        
        # Additional validation specific to confirmations
        assert is_valid_court_identifier(record['circuit_district']), \
            f"Invalid court identifier: {record['circuit_district']}"


def test_detect_table_format_modern():
    """Test that modern tables with thead/tbody are correctly identified."""
    # Modern table with thead and tbody
    modern_html = """
    <table>
        <thead>
            <tr>
                <th>Nominee</th>
                <th>Nomination Date</th>
                <th>Confirmation Date</th>
                <th>Circuit/District</th>
                <th>Incumbent</th>
                <th>Vacancy Date</th>
                <th>Vacancy Reason</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Doe, John</td>
                <td>01/01/2023</td>
                <td>02/01/2023</td>
                <td>01 - 1st Circuit</td>
                <td>Smith, Jane</td>
                <td>12/31/2022</td>
                <td>Retired</td>
            </tr>
        </tbody>
    </table>
    """
    soup = BeautifulSoup(modern_html, 'html.parser')
    table = soup.find('table')
    assert _detect_table_format(table) == 'modern'

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
    try:
        legacy_html = (fixtures_dir / "pages" / "2010" / "01" / "confirmations.html").read_text()
        soup = BeautifulSoup(legacy_html, 'html.parser')
        table = soup.find('table')
        if table:
            assert _detect_table_format(table) == 'legacy', "2010 fixture should be detected as legacy format"
    except FileNotFoundError:
        pytest.skip("2010 confirmations fixture not found")
    
    # Test with a known modern fixture (2024)
    try:
        modern_html = (fixtures_dir / "pages" / "2024" / "01" / "confirmations.html").read_text()
        soup = BeautifulSoup(modern_html, 'html.parser')
        table = soup.find('table')
        if table:
            assert _detect_table_format(table) == 'modern', "2024 fixture should be detected as modern format"
    except FileNotFoundError:
        pytest.skip("2024 confirmations fixture not found")
