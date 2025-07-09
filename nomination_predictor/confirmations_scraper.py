"""
Confirmations table extraction for judicial nominations data.

This module handles the extraction of confirmation data from HTML tables in the US Courts website.
It supports both legacy (pre-2016) and modern (2016+) table formats.
"""

import re
from typing import Any, Dict, List

from bs4 import BeautifulSoup


def is_valid_court_identifier(court: str) -> bool:
    """Check if a string is a valid court identifier.
    
    Args:
        court: The court identifier string to validate
        
    Returns:
        bool: True if the identifier matches known court formats, False otherwise
    """
    if not isinstance(court, str) or not court.strip():
        return False
        
    court = court.strip()
    return bool(
        re.match(r'^\d+\s*-\s*[A-Za-z]+', court) or  # 01 - CCA
        re.match(r'^[A-Za-z]+\s*-\s*[A-Za-z]+', court) or  # DC - DC, FD - CCA
        re.match(r'^IT$', court) or  # International Trade court
        re.match(r'^CL$', court) or  # Federal Claims court
        re.match(r'^SC$', court) or  # Supreme Court
        re.match(r'^[A-Za-z\s]+(?:Circuit|Court|District)', court, re.IGNORECASE)  # 1st Circuit
    )


def _get_column_indices(header_row: Any) -> Dict[str, int]:
    """Extract column indices from table header row.
    
    Args:
        header_row: BeautifulSoup row element containing header cells
        
    Returns:
        Dictionary mapping field names to column indices
    """
    headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
    
    # Map possible header names to our field names
    field_mapping = {
        'nominee': ['nominee'],
        'nomination_date': ['nomination date', 'nominated', 'date nominated'],
        'confirmation_date': ['confirmation date', 'confirmed', 'date confirmed'],
        'circuit_district': ['circuit/district', 'court', 'circuit', 'district'],
        'incumbent': ['incumbent'],
        'vacancy_date': ['vacancy date', 'date of vacancy'],
        'vacancy_reason': ['vacancy reason', 'reason']
    }
    
    # Initialize with default indices that will raise KeyError if not found
    indices = {}
    
    for field, possible_headers in field_mapping.items():
        for header in possible_headers:
            try:
                indices[field] = headers.index(header)
                break
            except ValueError:
                continue
    
    return indices

def _extract_modern_format(table: Any) -> List[Dict[str, str]]:
    """Extract data from modern (2016+) HTML format.
    
    Modern tables use thead/tbody structure and have consistent column ordering.
    """
    records: List[Dict[str, str]] = []
    
    # Find header row (usually in thead > tr > th)
    thead = table.find('thead')
    header_row = thead.find('tr') if thead else table.find('tr')
    
    if not header_row:
        return records  # No header row found
    
    # Get column indices from header
    try:
        col_indices = _get_column_indices(header_row)
    except (ValueError, KeyError):
        return records  # Couldn't determine column structure
    
    # Find all data rows (skip header)
    tbody = table.find('tbody')
    rows = tbody.find_all('tr') if tbody else table.find_all('tr')[1:]  # Skip header row
    
    for row in rows:
        if row.find('th'):  # Skip any remaining header rows
            continue
            
        cells = [cell.get_text(strip=True) for cell in row.find_all('td')]
        if not cells:
            continue
            
        try:
            # Extract data using column indices
            record = {
                'nominee': cells[col_indices['nominee']],
                'nomination_date': cells[col_indices['nomination_date']],
                'confirmation_date': cells[col_indices['confirmation_date']],
                'circuit_district': cells[col_indices['circuit_district']],
                'incumbent': cells[col_indices.get('incumbent', -1)] if col_indices.get('incumbent', -1) < len(cells) else '',
                'vacancy_date': cells[col_indices.get('vacancy_date', -1)] if col_indices.get('vacancy_date', -1) < len(cells) else '',
                'vacancy_reason': cells[col_indices.get('vacancy_reason', -1)] if col_indices.get('vacancy_reason', -1) < len(cells) else '',
            }
            
            # Only add the record if it has all required fields
            if all(record[field] for field in ['nominee', 'nomination_date', 'confirmation_date', 'circuit_district']):
                records.append(record)
                
        except (IndexError, KeyError):
            # Skip rows that don't match our expected format
            continue
    
    return records


def _is_valid_date(date_str: str) -> bool:
    """Check if a string is a valid date in MM/DD/YYYY format."""
    try:
        if not date_str or len(date_str) != 10 or date_str[2] != '/' or date_str[5] != '/':
            return False
        month, day, year = map(int, date_str.split('/'))
        if month < 1 or month > 12 or day < 1 or day > 31 or year < 1900 or year > 2100:
            return False
        return True
    except (ValueError, IndexError):
        return False

def _is_header_row(cells: List[str]) -> bool:
    """Check if a row appears to be a header row based on cell contents."""
    if not cells:
        return False
    
    header_indicators = [
        'nominee', 'nomination', 'date', 'confirmation',
        'circuit', 'district', 'incumbent', 'vacancy', 'reason'
    ]
    
    # Check if any cell contains header-like text
    cell_text = ' '.join(cells).lower()
    return any(indicator in cell_text for indicator in header_indicators)

def _extract_legacy_format(table: Any) -> List[Dict[str, str]]:
    """Extract data from legacy format tables (pre-2016).
    
    These tables don't use thead/tbody and have header rows with <strong> tags.
    They also have a footer row with 'Total Confirmations' that needs to be skipped.
    """
    records: List[Dict[str, str]] = []
    rows = table.find_all('tr')
    
    # Find the header row to determine where data starts
    header_row_idx = -1
    for i, row in enumerate(rows):
        if row.find('strong'):
            header_row_idx = i
            break
    
    if header_row_idx == -1:
        return []  # No header row found
    
    # Process data rows (skip header and any rows before it)
    for row in rows[header_row_idx + 1:]:
        # Skip rows that don't have any cells
        cells = [cell.get_text(strip=True) for cell in row.find_all('td')]
        if not cells:
            continue
            
        # Skip footer rows and header-like rows
        row_text = ' '.join(cells).lower()
        if ('total' in row_text and ('confirmation' in row_text or 'nomination' in row_text)) or _is_header_row(cells):
            continue
            
        # Need at least 4 cells for required fields
        if len(cells) < 4:
            continue
            
        try:
            # Extract data based on fixed column positions
            record = {
                'nominee': cells[0],
                'nomination_date': cells[1] if len(cells) > 1 else '',
                'confirmation_date': cells[2] if len(cells) > 2 else '',
                'circuit_district': cells[3] if len(cells) > 3 else '',
                'incumbent': cells[4] if len(cells) > 4 else '',
                'vacancy_date': cells[5] if len(cells) > 5 else '',
                'vacancy_reason': cells[6] if len(cells) > 6 else ''
            }
            
            # Skip if required fields are missing or empty
            required_fields = ['nominee', 'nomination_date', 'confirmation_date', 'circuit_district']
            if not all(record[field].strip() for field in required_fields):
                continue
                
            # Validate date formats
            if not _is_valid_date(record['nomination_date']) or not _is_valid_date(record['confirmation_date']):
                continue
                
            # All checks passed, add the record
            records.append(record)
                
        except (IndexError, KeyError, AttributeError):
            # Skip rows that cause errors during processing
            continue
    
    return records


def _detect_table_format(table: Any) -> str:
    """Detect the format of the table (modern or legacy).
    
    Modern tables typically have <thead> and <tbody> sections, while legacy tables
    have header rows with <strong> tags or specific text patterns.
    
    Args:
        table: BeautifulSoup table element
        
    Returns:
        str: 'legacy' or 'modern' based on table structure
    """
    # Check for modern table structure (thead/tbody)
    if table.find('thead') or table.find('tbody'):
        return 'modern'
    
    # Check for legacy header patterns in the first few rows
    header_keywords = ['nominee', 'court', 'position', 'nominated', 'confirmed']
    
    # Check first 5 rows for header patterns
    for row in table.find_all('tr', limit=5):
        # Check for header cells
        if row.find('th'):
            return 'modern'
            
        # Check for header text patterns
        row_text = row.get_text().lower()
        if any(keyword in row_text for keyword in header_keywords):
            return 'legacy'
    
    # Default to modern if we can't determine
    return 'modern'


def extract_confirmations_table(html: str) -> List[Dict[str, str]]:
    """
    Extract judicial confirmation data from month-level HTML content.

    Handles both older (pre-2016) and newer (2016+) HTML formats.

    Args:
        html: HTML content containing a table with judicial confirmation data

    Returns:
        List of dictionaries containing extracted data with standardized fields:
        - court: Name of the court/district
        - nominee: Name of the nominee
        - position: Position being confirmed
        - nominated: Date of nomination
        - hearing_date: Date of confirmation hearing
        - committee_vote: Date of committee vote
        - confirmed: Date of confirmation
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    
    if not table:
        return []
    
    # Determine table format and extract data accordingly
    table_format = _detect_table_format(table)
    if table_format == 'modern':
        records = _extract_modern_format(table)
    else:
        records = _extract_legacy_format(table)
    
    return records
