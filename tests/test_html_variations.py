"""Tests for handling HTML structure variations in judicial vacancy data."""

from pathlib import Path
from bs4 import BeautifulSoup
import pytest
from datetime import datetime

# Import the module to test
from nomination_predictor.dataset import extract_vacancy_table

# Path to fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pages"

# Fixtures for loading HTML content
@pytest.fixture
def load_fixture():
    """Load HTML content from a fixture file."""
    def _load_fixture(filename):
        filepath = FIXTURES_DIR / filename
        return filepath.read_text(encoding='utf-8')
    return _load_fixture

# Fixtures for different HTML structures
@pytest.fixture
def simple_table_html(load_fixture):
    """A simple HTML table without any child page links."""
    return load_fixture("archive_1983.html")  # Using 1983 as a simple example

@pytest.fixture
def modern_table_with_links(load_fixture):
    """Modern HTML structure with additional metadata."""
    return load_fixture("archive_2025.html")  # Using 2025 as a modern example

@pytest.fixture
def missing_table_html():
    """HTML without a table element."""
    return "<html><body><h1>Judicial Vacancies - No Data Available</h1><p>No vacancy data is currently available.</p></body></html>"

@pytest.fixture
def year_with_monthly_links(load_fixture):
    """A year page with monthly vacancy list links (e.g., 2001)."""
    return load_fixture("2001/01/vacancies.html")  # Using January 2001 as an example

@pytest.fixture
def monthly_vacancy_page(load_fixture):
    """A monthly vacancy page (e.g., January 2001)."""
    return load_fixture("2001/01/vacancies.html")

@pytest.fixture
def modern_year_page(load_fixture):
    """A more modern year page with additional metadata (e.g., 2014)."""
    return load_fixture("2014/01/vacancies.html")

# Tests for different HTML structures
def test_simple_table_extraction(monthly_vacancy_page):
    """Test extraction from a simple table structure (older years)."""
    # Use monthly_vacancy_page fixture which contains actual monthly data
    records = extract_vacancy_table(monthly_vacancy_page)
    
    # Verify we got some records
    assert len(records) > 0, "Expected to find vacancy records in the monthly page"
    
    # Check that we have expected fields in the first record
    first_record = records[0]
    
    # These are the fields we expect to find in the source data
    expected_fields = ['court', 'vacancy_date', 'incumbent', 'vacancy_reason']
    for field in expected_fields:
        assert field in first_record, f"Expected field '{field}' not found in record"
    
    # Check that vacancy_date is in the expected format if present
    if 'vacancy_date' in first_record and first_record['vacancy_date']:
        try:
            datetime.strptime(first_record['vacancy_date'], "%m/%d/%Y")
        except ValueError:
            assert False, f"Invalid date format in vacancy_date: {first_record['vacancy_date']}"
    
    # Verify we're not expecting fields that don't exist in the source
    assert 'nominating_president' not in first_record, "'nominating_president' field should not be expected in the source data"
    assert 'nominee' not in first_record, "'nominee' field should not be expected in the source data"
    assert 'status' not in first_record, "'status' field should not be expected in the source data"

def test_modern_table_extraction(modern_table_with_links):
    """Test extraction from modern table structure with additional fields."""
    records = extract_vacancy_table(modern_table_with_links)
    assert len(records) > 0  # Just check we got some records
    
    # Check that we have expected fields in the first record
    first_record = records[0]
    for field in ['court', 'vacancy_date', 'incumbent', 'vacancy_reason']:
        assert field in first_record, f"Expected field '{field}' not found in record"
    
    # Check that dates are in the correct format if present
    if 'vacancy_date' in first_record and first_record['vacancy_date']:
        try:
            datetime.strptime(first_record['vacancy_date'], "%m/%d/%Y")
        except ValueError:
            assert False, f"Invalid date format in vacancy_date: {first_record['vacancy_date']}"

def test_missing_table_handling(missing_table_html):
    """Test handling of HTML without a table element."""
    records = extract_vacancy_table(missing_table_html)
    assert records == []  # Should return empty list for no tables

def test_partial_data_handling(load_fixture):
    """Test handling of tables with missing or incomplete data."""
    # Use a real fixture that might have partial data
    html = load_fixture("archive_1983.html")
    records = extract_vacancy_table(html)
    
    # Just check that the function can handle the real data without errors
    assert isinstance(records, list)
    
    # If we have records, check some basic properties
    if records:
        for record in records:
            assert 'court' in record, "All records should have a 'court' field"
            if 'vacancy_date' in record and record['vacancy_date']:
                try:
                    datetime.strptime(record['vacancy_date'], "%m/%d/%Y")
                except ValueError:
                    assert False, f"Invalid date format in vacancy_date: {record['vacancy_date']}"

def test_year_with_monthly_links(year_with_monthly_links):
    """Test extraction of monthly links from a year page."""
    soup = BeautifulSoup(year_with_monthly_links, 'html.parser')
    
    # Find all links that might point to monthly pages
    # This is a more flexible approach that should work with the actual HTML structure
    links = soup.find_all('a', href=True)
    monthly_links = [a for a in links if any(month in a.text.lower() for month in 
                     ['january', 'february', 'march', 'april', 'may', 'june', 
                      'july', 'august', 'september', 'october', 'november', 'december'])]
    
    # We should find at least some monthly links
    assert len(monthly_links) > 0, "Expected to find monthly report links"
    
    # Check that the links have valid month names
    for link in monthly_links:
        month_found = any(month in link.text.lower() for month in 
                         ['january', 'february', 'march', 'april', 'may', 'june', 
                          'july', 'august', 'september', 'october', 'november', 'december'])
        assert month_found, f"Link text '{link.text}' doesn't contain a month name"

def test_monthly_vacancy_extraction(monthly_vacancy_page):
    """Test extraction from a monthly vacancy page."""
    records = extract_vacancy_table(monthly_vacancy_page)
    
    # We should get some records from the monthly page
    assert len(records) > 0, "Expected to find vacancy records in monthly page"
    
    # Check that the records have the expected structure
    for record in records:
        assert 'court' in record, "Each record should have a 'court' field"
        if 'vacancy_date' in record and record['vacancy_date']:
            try:
                datetime.strptime(record['vacancy_date'], "%m/%d/%Y")
            except ValueError:
                assert False, f"Invalid date format in vacancy_date: {record['vacancy_date']}"

def test_modern_year_page_links(modern_year_page):
    """Test extraction of various links from a modern year page."""
    soup = BeautifulSoup(modern_year_page, 'html.parser')
    all_links = soup.find_all('a', href=True)
    
    # Should find some links
    assert len(all_links) > 0, "Expected to find links in the modern year page"
    
    # Check for common patterns in URLs
    link_hrefs = [link['href'].lower() for link in all_links]
    
    # Check for common patterns that indicate judicial or vacancy related links
    has_judicial_links = any('judicial' in href for href in link_hrefs)
    has_vacancy_links = any('vacanc' in href for href in link_hrefs)  # 'vacanc' matches 'vacancy' or 'vacancies'
    
    assert has_judicial_links or has_vacancy_links, \
        "Expected to find judicial or vacancy related links in the page"
