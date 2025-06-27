"""Tests for handling HTML structure variations in judicial vacancy data."""

from pathlib import Path

from bs4 import BeautifulSoup

# Import the module to test
from data import make_dataset
import pytest


# Fixtures for different HTML structures
@pytest.fixture
def simple_table_html():
    """A simple HTML table without any child page links."""
    return """
    <html>
      <body>
        <h1>Judicial Vacancies - May 1983</h1>
        <table>
          <tr><th>Court</th><th>Vacancy Date</th><th>Status</th></tr>
          <tr><td>9th Circuit</td><td>01/01/1983</td><td>Vacant</td></tr>
        </table>
      </body>
    </html>
    """

@pytest.fixture
def modern_table_with_links():
    """Modern HTML structure with additional metadata."""
    return """
    <html>
      <head>
        <title>Judicial Vacancies - March 2025 | United States Courts</title>
      </head>
      <body>
        <div class="main-content">
          <h1>Judicial Vacancies - March 2025</h1>
          <div class="vacancy-summary">
            <p>Showing vacancies as of March 1, 2025</p>
          </div>
          <table class="responsive-table">
            <thead>
              <tr>
                <th>Court</th>
                <th>Vacancy Date</th>
                <th>Nominating President</th>
                <th>Nominee</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>9th Circuit</td>
                <td>01/15/2025</td>
                <td>Biden</td>
                <td>John Smith</td>
                <td>Pending Hearing</td>
              </tr>
              <tr>
                <td>DC Circuit</td>
                <td>02/20/2025</td>
                <td>Biden</td>
                <td>Jane Doe</td>
                <td>Pending Committee Vote</td>
              </tr>
            </tbody>
          </table>
          <div class="pagination">
            <span>1-2 of 2 results</span>
          </div>
        </div>
      </body>
    </html>
    """

@pytest.fixture
def missing_table_html():
    """HTML without a table element."""
    return """
    <html>
      <body>
        <h1>Judicial Vacancies - No Data Available</h1>
        <p>No vacancy data is currently available.</p>
      </body>
    </html>
    """

@pytest.fixture
def year_with_monthly_links():
    """A year page with monthly vacancy list links (e.g., 1981)."""
    return """
    <html>
      <head>
        <title>Judicial Vacancies - 1981 | United States Courts</title>
      </head>
      <body>
        <div class="main-content">
          <h1>Judicial Vacancies - 1981</h1>
          <div class="archive-links">
            <h2>Monthly Reports</h2>
            <ul>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/1981/01">January 1981</a></li>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/1981/02">February 1981</a></li>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/1981/03">March 1981</a></li>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/1981/04">April 1981</a></li>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/1981/05">May 1981</a></li>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/1981/06">June 1981</a></li>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/1981/07">July 1981</a></li>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/1981/08">August 1981</a></li>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/1981/09">September 1981</a></li>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/1981/10">October 1981</a></li>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/1981/11">November 1981</a></li>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/1981/12">December 1981</a></li>
            </ul>
          </div>
        </div>
      </body>
    </html>
    """

@pytest.fixture
def monthly_vacancy_page():
    """A monthly vacancy page from an early year (e.g., January 1981)."""
    return """
    <html>
      <head>
        <title>Judicial Vacancies - January 1981 | United States Courts</title>
      </head>
      <body>
        <div class="main-content">
          <h1>Judicial Vacancies - January 1981</h1>
          <div class="vacancy-summary">
            <p>Showing vacancies as of January 1, 1981</p>
          </div>
          <table class="responsive-table">
            <thead>
              <tr>
                <th>Court</th>
                <th>Vacancy Date</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>D. Alaska</td>
                <td>01/01/1981</td>
                <td>Vacant</td>
              </tr>
              <tr>
                <td>D. Arizona</td>
                <td>01/15/1981</td>
                <td>Nominated: John Smith</td>
              </tr>
            </tbody>
          </table>
        </div>
      </body>
    </html>
    """

@pytest.fixture
def modern_year_page():
    """A modern year page with multiple types of links (e.g., 2025)."""
    return """
    <html>
      <head>
        <title>Judicial Vacancies - 2025 | United States Courts</title>
      </head>
      <body>
        <div class="main-content">
          <h1>Judicial Vacancies - 2025</h1>
          <div class="archive-links">
            <h2>Reports</h2>
            <ul>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/2025/01">January 2025</a></li>
              <li><a href="/judges-judgeships/archives/judicial-vacancies/2025/02">February 2025</a></li>
            </ul>
            <h3>Additional Resources</h3>
            <ul>
              <li><a href="/judges-judgeships/vacancies/emergencies">Judicial Emergencies</a></li>
              <li><a href="/judges-judgeships/vacancies/confirmations">Pending Confirmations</a></li>
              <li><a href="/judges-judgeships/vacancies/summary">Yearly Summary</a></li>
            </ul>
          </div>
        </div>
      </body>
    </html>
    """

# Tests for different HTML structures
def test_simple_table_extraction(simple_table_html):
    """Test extraction from a simple table structure (older years)."""
    records = make_dataset.extract_vacancy_table(simple_table_html)
    assert len(records) == 1
    assert records[0]["court"] == "9th Circuit"
    assert records[0]["vacancy_date"] == "01/01/1983"
    assert records[0]["status"] == "Vacant"
    # These fields should be None or empty for older records
    assert records[0].get("nominating_president") is None
    assert records[0].get("nominee") is None

def test_modern_table_extraction(modern_table_with_links):
    """Test extraction from modern table structure with additional fields."""
    records = make_dataset.extract_vacancy_table(modern_table_with_links)
    
    # Should find 2 records
    assert len(records) == 2
    
    # Check first record
    assert records[0]["court"] == "9th Circuit"
    assert records[0]["vacancy_date"] == "01/15/2025"
    assert records[0]["nominating_president"] == "Biden"
    assert records[0]["nominee"] == "John Smith"
    assert records[0]["status"] == "Pending Hearing"
    
    # Check second record
    assert records[1]["court"] == "DC Circuit"
    assert records[1]["vacancy_date"] == "02/20/2025"
    assert records[1]["nominating_president"] == "Biden"
    assert records[1]["nominee"] == "Jane Doe"
    assert records[1]["status"] == "Pending Committee Vote"

def test_missing_table_handling(missing_table_html):
    """Test handling of HTML without a table element."""
    records = make_dataset.extract_vacancy_table(missing_table_html)
    assert records == []  # Should return empty list for no tables

def test_partial_data_handling():
    """Test handling of tables with missing or incomplete data."""
    html = """
    <html>
      <body>
        <table>
          <tr><th>Court</th><th>Vacancy Date</th></tr>
          <tr><td>9th Circuit</td><td>01/01/2023</td></tr>
          <tr><td>DC Circuit</td><td></td></tr>
          <tr><td></td><td>01/01/2024</td></tr>
        </table>
      </body>
    </html>
    """
    records = make_dataset.extract_vacancy_table(html)
    assert len(records) == 3
    # First record should be complete
    assert records[0]["court"] == "9th Circuit"
    assert records[0]["vacancy_date"] == "01/01/2023"
    # Second record missing date
    assert records[1]["court"] == "DC Circuit"
    assert records[1]["vacancy_date"] == ""
    # Third record missing court
    assert records[2]["court"] == ""
    assert records[2]["vacancy_date"] == "01/01/2024"

def test_year_with_monthly_links(year_with_monthly_links):
    """Test extraction of monthly links from a year page."""
    soup = BeautifulSoup(year_with_monthly_links, 'html.parser')
    links = soup.select('.archive-links a')
    
    # Should find 12 monthly links
    assert len(links) == 12
    
    # Check first and last links
    assert links[0]['href'] == "/judges-judgeships/archives/judicial-vacancies/1981/01"
    assert links[0].text.strip() == "January 1981"
    assert links[-1]['href'] == "/judges-judgeships/archives/judicial-vacancies/1981/12"
    assert links[-1].text.strip() == "December 1981"

def test_monthly_vacancy_extraction(monthly_vacancy_page):
    """Test extraction from a monthly vacancy page (early years)."""
    records = make_dataset.extract_vacancy_table(monthly_vacancy_page)
    
    assert len(records) == 2
    
    # First vacancy
    assert records[0]["court"] == "D. Alaska"
    assert records[0]["vacancy_date"] == "01/01/1981"
    assert records[0]["status"] == "Vacant"
    
    # Second vacancy
    assert records[1]["court"] == "D. Arizona"
    assert records[1]["vacancy_date"] == "01/15/1981"
    assert records[1]["status"] == "Nominated: John Smith"

def test_modern_year_page_links(modern_year_page):
    """Test extraction of various links from a modern year page."""
    soup = BeautifulSoup(modern_year_page, 'html.parser')
    
    # Get all links in the archive-links section
    all_links = soup.select('.archive-links a')
    monthly_links = [a for a in all_links if 'judicial-vacancies' in a['href']]
    resource_links = [a for a in all_links if 'judicial-vacancies' not in a['href']]
    
    # Should find 2 monthly links and 3 resource links
    assert len(monthly_links) == 2
    assert len(resource_links) == 3
    
    # Check the first monthly link
    assert monthly_links[0]['href'] == "/judges-judgeships/archives/judicial-vacancies/2025/01"
    assert monthly_links[0].text.strip() == "January 2025"
    
    # Check one of the resource links
    assert any(link['href'] == "/judges-judgeships/vacancies/emergencies" for link in resource_links)
    assert any(link.text.strip() == "Pending Confirmations" for link in resource_links)
