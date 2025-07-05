"""Tests for handling HTML structure variations in judicial vacancy data.

This test suite verifies the extraction of vacancy data from different page types:
1. Year-level pages (e.g., archive_1983.html, archive_2025.html) - Contain links to monthly reports
2. Month-level pages (e.g., 2001/01/vacancies.html) - Contain detailed vacancy information
3. ignores as out-of-scope any PDF documents (e.g., 1983/01/vacancies.pdf)

Each test is clearly labeled to indicate the input type and expected behavior.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict

import pytest

# Import the module to test
from nomination_predictor.dataset import extract_vacancy_table, generate_month_links

# Path to fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pages"

# Format Analysis Results
"""
Document Format Analysis:

A. Year-Level Pages (e.g., archive_YYYY.html):
   - archive_1983.html: Legacy HTML format (contains month links)
   - archive_2025.html: Modern HTML format (contains month links)
   Expected output: List of dictionaries with 'url', 'month', and 'year' keys

B. Month-Level Pages:
   1. HTML Format (e.g., 2001/01/vacancies.html):
      - Standardized format with detailed vacancy information
   2. PDF Formats:
      - 1981-2001/11: Older PDF format
      - 2001/12-2009/06: Newer PDF format with different layout
      Note: All PDF formats are currently unsupported and may raise ParseError
"""

# Fixtures for loading test data
@pytest.fixture
def load_fixture():
    """Load content from a fixture file with proper error handling."""
    def _load_fixture(filename):
        filepath = FIXTURES_DIR / filename
        return filepath.read_text(encoding='utf-8', errors='replace')
    return _load_fixture

# Year-Level Page Fixtures
@pytest.fixture
def year_2025_modern_page(load_fixture):
    """Year-level page with modern HTML structure (2025)."""
    return load_fixture("archive_2025.html")

# Month-Level Page Fixtures (HTML)
@pytest.fixture
def month_2010_01_page(load_fixture):
    """Month-level HTML page (January 2010)."""
    return load_fixture("2010/01/vacancies.html")


# Test Utilities
def validate_month_link(link: Dict[str, str]) -> None:
    """Validate that a month link has the expected structure and data types."""
    assert isinstance(link, dict), "Month link should be a dictionary"
    assert 'url' in link, "Month link is missing 'url' field"
    assert 'month' in link, "Month link is missing 'month' field"
    assert isinstance(link['month'], str), "Month name should be a string"
    assert link['month'] in [
        '01', '02', '03', '04', '05', '06',
        '07', '08', '09', '10', '11', '12'
    ], f"Invalid month number: {link['month']}"
    assert 'year' in link, "Month link is missing 'year' field"
    assert isinstance(link['year'], str), "Year should be a string"

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

# Year-Level Page Tests

def test_generate_month_links_from_year_2025_modern_page(year_2025_modern_page):
    """Test generation of month links from year-level page with modern HTML structure (2025)."""
    month_links = generate_month_links(year_2025_modern_page)
    
    # Verify we got some month links
    assert len(month_links) > 0, "Expected to find month links in the 2025 year page"
    
    # Validate each month link
    for link in month_links:
        validate_month_link(link)
        

# Month-Level HTML Page Tests
def test_extract_vacancies_from_month_2010_01_page(month_2010_01_page):
    """Test extraction of vacancy data from month-level HTML page (January 2010)."""
    records = extract_vacancy_table(month_2010_01_page)
    assert len(records) > 0, "Expected to find vacancy records in the January 2010 month page"
    
    for record in records:
        validate_vacancy_record(record)
