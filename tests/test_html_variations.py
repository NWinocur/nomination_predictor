"""Tests for handling HTML structure variations in judicial vacancy data."""

from datetime import datetime
from pathlib import Path
from typing import Dict

import pytest

# Import the module to test
from nomination_predictor.dataset import ParseError, extract_vacancy_table

# Path to fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pages"

# Format Analysis Results
"""
Based on analysis of the fixtures, here are the different formats found:

1. PDF Formats:
   - 1981-2001/12: Older PDF format (e.g., 1983/01/vacancies.pdf)
   - 2001/12: Newer PDF format with different layout (vacancies.pdf)
   - TODO: add support for PDFs so that known/expected formats are processed without a raised ParseError

2. HTML Formats:
   - archive_1983.html: Legacy HTML format (simple table)
   - archive_2025.html: Modern HTML format (more structured with additional metadata)
   - 2001/01/vacancies.html: Monthly report format (different structure from yearly archives)
"""

# Fixtures for loading HTML content
@pytest.fixture
def load_fixture():
    """Load content from a fixture file."""
    def _load_fixture(filename):
        filepath = FIXTURES_DIR / filename
        return filepath.read_text(encoding='utf-8', errors='replace')
    return _load_fixture

# Fixtures for different HTML structures
@pytest.fixture
def simple_table_html(load_fixture):
    """A simple HTML table without any child page links."""
    return load_fixture("archive_1983.html")  # Legacy HTML format

@pytest.fixture
def modern_table_with_links(load_fixture):
    """Modern HTML structure with additional metadata."""
    return load_fixture("archive_2025.html")  # Modern HTML format

@pytest.fixture
def missing_table_html():
    """HTML without a table element."""
    return "<html><body><h1>Judicial Vacancies - No Data Available</h1><p>No vacancy data is currently available.</p></body></html>"

@pytest.fixture
def monthly_vacancy_page(load_fixture):
    """A monthly vacancy page (e.g., January 2001)."""
    return load_fixture("2001/01/vacancies.html")

@pytest.fixture
def legacy_pdf_vacancy_page():
    """A legacy PDF vacancy page (e.g., 1983/01/vacancies.pdf)."""
    path = FIXTURES_DIR / "1983" / "01" / "vacancies.pdf"
    return path.read_bytes()

@pytest.fixture
def dec_2001_pdf_vacancy_page():
    """December 2001 PDF vacancy page with different format (2001/12/vacancies.pdf)."""
    path = FIXTURES_DIR / "2001" / "12" / "vacancies.pdf"
    return path.read_bytes()

# Helper functions
def validate_vacancy_record(record: Dict[str, str]) -> None:
    """Validate that a vacancy record has the expected structure and data types."""
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

# Tests for different HTML structures
def test_legacy_table_extraction(simple_table_html):
    """Test extraction from legacy HTML table structure (1980s-1990s)."""
    records = extract_vacancy_table(simple_table_html)
    assert len(records) > 0, "Expected to find vacancy records in the legacy HTML"
    
    for record in records:
        validate_vacancy_record(record)
    
    first_record = records[0]
    assert all(field in first_record for field in ['court', 'vacancy_date', 'incumbent', 'vacancy_reason'])

def test_modern_table_extraction(modern_table_with_links):
    """Test extraction from modern HTML table structure with additional fields."""
    records = extract_vacancy_table(modern_table_with_links)
    assert len(records) > 0, "Expected to find vacancy records in the modern HTML"
    
    for record in records:
        validate_vacancy_record(record)
    
    first_record = records[0]
    assert any(field in first_record for field in ['nominee', 'nomination_date', 'confirmation_date'])

def test_missing_table_handling(missing_table_html):
    """Test handling of HTML without a table element."""
    records = extract_vacancy_table(missing_table_html)
    # TODO: make this test that the extract_vacancy_table() function under test raises some manner of error if no table is found for the input specified
    assert records == []

def test_monthly_vacancy_extraction(monthly_vacancy_page):
    """Test extraction from a monthly vacancy page."""
    records = extract_vacancy_table(monthly_vacancy_page)
    assert len(records) > 0, "Expected to find vacancy records in monthly page"
    
    for record in records:
        validate_vacancy_record(record)

# PDF Format Tests
def test_legacy_pdf_error_handling(legacy_pdf_vacancy_page):
    """Test error handling for legacy PDF format (1981-2001/11)."""
    with pytest.raises(ParseError) as excinfo:
        extract_vacancy_table(legacy_pdf_vacancy_page.decode('latin-1'))
    # TODO: replace this test case with expectation of proper PDF support
    assert "PDF format is not currently supported" in str(excinfo.value)

def test_dec_2001_pdf_error_handling(dec_2001_pdf_vacancy_page):
    """Test error handling for December 2001 PDF format."""
    with pytest.raises(ParseError) as excinfo:
        extract_vacancy_table(dec_2001_pdf_vacancy_page.decode('latin-1'))
    # TODO: replace this test case with expectation of proper PDF support
    assert "PDF format is not currently supported" in str(excinfo.value)

# Format Analysis Test
def test_pdf_formats_consistency():
    """Verify that all PDF files are properly handled (currently all raise ParseError)."""
    # This test ensures we don't accidentally start supporting PDFs without proper test coverage
    # TODO: replace this test case with expectation of proper PDF support
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
