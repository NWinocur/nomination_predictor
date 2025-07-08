"""
Emergency declarations table extraction for judicial vacancies data.

This module handles the extraction of emergency declaration data from HTML tables
in the US Courts website. It supports both legacy (pre-2016) and modern (2016+)
table formats for emergency declarations.
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
        'circuit_district': ['circuit/district', 'circuit', 'district'],
        'title': ['title', 'position'],
        'vacancy_judge': ['vacancy created by', 'judge', 'vacancy judge'],
        'reason': ['reason'],
        'vacancy_date': ['vacancy date', 'date'],
        'days_pending': ['days pending', 'pending'],
        'weighted': ['weighted*', 'weighted'],
        'adjusted': ['adjusted*', 'adjusted']
    }
    
    indices = {}
    for field, possible_headers in field_mapping.items():
        for i, header in enumerate(headers):
            if header and any(h in header.lower() for h in possible_headers):
                indices[field] = i
                break
    
    return indices


def _extract_modern_format(table: Any) -> List[Dict[str, str]]:
    """Extract data from modern (2016+) HTML format.
    
    Modern tables use thead/tbody structure and have consistent column ordering.
    """
    records = []
    thead = table.find('thead')
    tbody = table.find('tbody')
    
    if not thead or not tbody:
        return []
    
    # Get column indices from header row
    header_row = thead.find('tr')
    if not header_row:
        return []
    
    column_indices = _get_column_indices(header_row)
    if not column_indices:
        return []
    
    # Process each data row
    for row in tbody.find_all('tr', recursive=False):
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        if not cells or _is_header_row(cells):
            continue
            
        record = {}
        for field, idx in column_indices.items():
            if idx < len(cells):
                record[field] = cells[idx]
        
        # Only add record if it has required fields
        if all(field in record for field in ['circuit_district', 'title', 'vacancy_judge', 'vacancy_date']):
            records.append(record)
    
    return records


def _is_valid_date(date_str: str) -> bool:
    """Check if a string is a valid date in MM/DD/YYYY format."""
    if not date_str or not isinstance(date_str, str):
        return False
    return bool(re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_str.strip()))


def _is_header_row(cells: List[str]) -> bool:
    """Check if a row appears to be a header row based on cell contents."""
    if not cells or len(cells) < 3:
        return False
    
    # Check for header-like content in first few cells
    header_indicators = ['court', 'judge', 'date', 'declaration', 'termination']
    first_cells = ' '.join(cell.lower() for cell in cells[:4] if cell)
    return any(indicator in first_cells for indicator in header_indicators)


def _extract_legacy_format(table: Any) -> List[Dict[str, str]]:
    """Extract data from legacy format tables (pre-2016).
    
    These tables don't use thead/tbody and may have inconsistent formatting.
    """
    records = []
    rows = table.find_all('tr', recursive=False)
    
    # Find the header row (usually the first non-empty row with header-like content)
    header_row_idx = None
    for i, row in enumerate(rows):
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        if _is_header_row(cells):
            header_row_idx = i
            break
    
    if header_row_idx is None:
        return []
    
    # Get column indices from header row
    header_row = rows[header_row_idx]
    column_indices = _get_column_indices(header_row)
    
    if not column_indices:
        return []
    
    # Process data rows (rows after header)
    for row in rows[header_row_idx + 1:]:
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        if not cells or _is_header_row(cells):
            continue
            
        record = {}
        for field, idx in column_indices.items():
            if idx < len(cells):
                record[field] = cells[idx]
        
        # Only add record if it has required fields and valid court
        if (all(field in record for field in ['circuit_district', 'title', 'vacancy_judge', 'vacancy_date']) and 
            is_valid_court_identifier(record['circuit_district'])):
            records.append(record)
    
    return records


def _detect_table_format(table: Any) -> str:
    """Detect the format of the table (modern or legacy).
    
    Modern tables typically have <thead> and <tbody> sections, while legacy tables
    have header rows with specific text patterns.
    
    Args:
        table: BeautifulSoup table element
        
    Returns:
        str: 'legacy' or 'modern' based on table structure
    """
    if not table:
        return 'modern'  # Default to modern for empty tables
    
    # Check for modern table structure
    if table.find('thead') and table.find('tbody'):
        return 'modern'
    
    # Check for legacy table structure
    rows = table.find_all('tr', recursive=False)
    if not rows:
        return 'modern'  # Default to modern for empty tables
    
    # Check first few rows for header-like content
    for row in rows[:min(3, len(rows))]:
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        if _is_header_row(cells):
            return 'legacy'
    
    # Default to modern if we can't determine
    return 'modern'


def extract_emergencies_table(html: str) -> List[Dict[str, str]]:
    """
    Extract judicial emergency declaration data from month-level HTML content.

    Handles both older (pre-2016) and newer (2016+) HTML formats.

    Args:
        html: HTML content containing a table with judicial emergency data

    Returns:
        List of dictionaries containing extracted data with standardized fields:
        - court: Name of the court/district
        - emergency_judge: Name of the judge who declared the emergency
        - emergency_declaration_date: Date when emergency was declared
        - emergency_termination_date: Date when emergency was terminated (if applicable)
        - emergency_extension: Information about any extensions (if applicable)
        - notes: Additional notes about the emergency (if any)
    """
    if not html or not isinstance(html, str):
        return []
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return []
        
        # Determine table format and extract data accordingly
        table_format = _detect_table_format(table)
        if table_format == 'modern':
            return _extract_modern_format(table)
        else:
            return _extract_legacy_format(table)
            
    except Exception as e:
        # Log the error but don't fail the entire process
        import logging
        logging.exception(f"Error extracting emergency data: {e}")
        return []
