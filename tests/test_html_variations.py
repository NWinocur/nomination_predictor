"""Tests for handling HTML structure variations in judicial vacancy data.

This test suite verifies the extraction of vacancy data from different page types:
1. Year-level pages (e.g., archive_1983.html, archive_2025.html) - Contain links to monthly reports
2. Month-level pages (e.g., 2001/01/vacancies.html) - Contain detailed vacancy information
3. PDF documents (e.g., 1983/01/vacancies.pdf) - Currently unsupported format

Each test is clearly labeled to indicate the input type and expected behavior.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict

import pytest

# Import the module to test
from nomination_predictor.dataset import ParseError, extract_month_links, extract_vacancy_table

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
      - 2001/12: Newer PDF format with different layout
      Note: All PDF formats are currently unsupported and should raise ParseError
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
def year_1983_legacy_page(load_fixture):
    """Year-level page with legacy HTML structure (1983)."""
    return load_fixture("archive_1983.html")

@pytest.fixture
def year_2025_modern_page(load_fixture):
    """Year-level page with modern HTML structure (2025)."""
    return load_fixture("archive_2025.html")

# Month-Level Page Fixtures (HTML)
@pytest.fixture
def month_2001_01_page(load_fixture):
    """Month-level HTML page (January 2001)."""
    return load_fixture("2001/01/vacancies.html")

# Month-Level Page Fixtures (PDF - Unsupported)
@pytest.fixture
def month_1983_01_pdf():
    """Month-level PDF page (January 1983) - Unsupported format."""
    path = FIXTURES_DIR / "1983" / "01" / "vacancies.pdf"
    return path.read_bytes()

@pytest.fixture
def month_2001_12_pdf():
    """Month-level PDF page (December 2001) - Newer format, still unsupported."""
    path = FIXTURES_DIR / "2001" / "12" / "vacancies.pdf"
    return path.read_bytes()

# Test Utilities
def validate_month_link(link: Dict[str, str]) -> None:
    """Validate that a month link has the expected structure and data types."""
    assert isinstance(link, dict), "Month link should be a dictionary"
    assert 'url' in link, "Month link is missing 'url' field"
    assert 'month' in link, "Month link is missing 'month' field"
    assert isinstance(link['month'], str), "Month name should be a string"
    assert link['month'].lower() in [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    ], f"Invalid month name: {link['month']}"

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
def test_extract_month_links_from_year_1983_legacy_page(year_1983_legacy_page):
    """Test extraction of month links from year-level page with legacy HTML structure (1983)."""
    month_links = extract_month_links(year_1983_legacy_page)
    
    # Verify we got some month links
    assert len(month_links) > 0, "Expected to find month links in the 1983 year page"
    
    # Validate each month link
    for link in month_links:
        validate_month_link(link)
        
    # Check that we have expected months (at least some of them)
    month_names = {link['month'].lower() for link in month_links}
    assert any(month in month_names for month in ['january', 'february', 'march']), \
        "Expected to find at least some common months in the links"

def test_extract_month_links_from_year_2025_modern_page(year_2025_modern_page):
    """Test extraction of month links from year-level page with modern HTML structure (2025)."""
    month_links = extract_month_links(year_2025_modern_page)
    
    # Verify we got some month links
    assert len(month_links) > 0, "Expected to find month links in the 2025 year page"
    
    # Validate each month link
    for link in month_links:
        validate_month_link(link)
        
    # Check that we have expected months (at least some of them)
    month_names = {link['month'].lower() for link in month_links}
    assert any(month in month_names for month in ['january', 'february', 'march']), \
        "Expected to find at least some common months in the links"

# Month-Level HTML Page Tests
def test_extract_vacancies_from_month_2001_01_page(month_2001_01_page):
    """Test extraction of vacancy data from month-level HTML page (January 2001)."""
    records = extract_vacancy_table(month_2001_01_page)
    assert len(records) > 0, "Expected to find vacancy records in the January 2001 month page"
    
    for record in records:
        validate_vacancy_record(record)

# Month-Level PDF Page Tests (Unsupported)
def test_handle_legacy_pdf_month_page(month_1983_01_pdf):
    """Test error handling for unsupported legacy PDF month page (January 1983)."""
    with pytest.raises(ParseError) as excinfo:
        extract_vacancy_table(month_1983_01_pdf.decode('latin-1'))
    assert "PDF format is not currently supported" in str(excinfo.value)

def test_handle_2001_dec_pdf_month_page(month_2001_12_pdf):
    """Test error handling for unsupported December 2001 PDF month page."""
    with pytest.raises(ParseError) as excinfo:
        extract_vacancy_table(month_2001_12_pdf.decode('latin-1'))
    assert "PDF format is not currently supported" in str(excinfo.value)

# Special Case Tests
def test_handle_empty_input():
    """Test handling of empty or invalid input."""
    with pytest.raises(ParseError):
        extract_vacancy_table("")
    
    with pytest.raises(ParseError):
        extract_vacancy_table("<html><body>No table here</body></html>")

def test_verify_all_pdf_formats_unsupported():
    """Verify that all PDF files consistently raise ParseError."""
    pdf_files = list(FIXTURES_DIR.glob('**/*.pdf'))
    assert len(pdf_files) > 0, "Expected to find PDF fixtures"
    
    for pdf_path in pdf_files:
        try:
            content = pdf_path.read_bytes().decode('latin-1')
            with pytest.raises(ParseError) as excinfo:
                extract_vacancy_table(content)
            assert "PDF format is not currently supported" in str(excinfo.value)
        except Exception as e:
            assert False, f"Unexpected error processing {pdf_path}: {str(e)}"
