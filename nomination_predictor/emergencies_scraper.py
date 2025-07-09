"""
Emergency declarations table extraction for judicial vacancies data.

This module handles the extraction of emergency declaration data from HTML tables
in the US Courts website. It supports both legacy (pre-2016) and modern (2016+)
table formats for emergency declarations.
"""

import re
from typing import Any, Dict, List

from bs4 import BeautifulSoup
from loguru import logger


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
    # Get all header cells, handling both th and td elements
    header_cells = header_row.find_all(['th', 'td'])
    
    # Extract text from each header cell, handling nested elements and cleaning up whitespace
    headers = []
    for cell in header_cells:
        # Get text content and clean it up
        text = cell.get_text(' ', strip=True).lower()
        # Remove any extra whitespace and normalize
        text = ' '.join(text.split())
        headers.append(text)
    
    # Map possible header names to our field names
    field_mapping = {
        'circuit_district': ['circuit/district', 'circuit', 'district', 'court'],
        'title': ['title', 'position'],
        'vacancy_created_by': ['vacancy created by', 'judge', 'vacancy judge', 'incumbent'],
        'reason': ['reason'],
        'vacancy_date': ['vacancy date', 'date', 'opening date'],
        'days_pending': ['days pending', 'pending', 'days'],
        'weighted': ['weighted*', 'weighted', 'weighted filings per judgeship*'],
        'adjusted': ['adjusted*', 'adjusted', 'adjusted filings per panel*']
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
    records: List[Dict[str, str]] = []
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
        if all(field in record for field in ['circuit_district', 'vacancy_date', 'days_pending']):
            records.append(record)
    
    return records

def _is_header_row(cells: List[Any]) -> bool:
    """Check if a row appears to be a header row based on cell contents and structure.
    
    Args:
        cells: List of BeautifulSoup cell elements (td/th)
        
    Returns:
        bool: True if the row appears to be a header row
    """
    if not cells or len(cells) < 3:
        return False
    
    def get_cell_text(cell):
        if hasattr(cell, 'get_text'):
            return cell.get_text(strip=True).lower()
        return str(cell).lower() if cell else ""
    
    cell_texts = [get_cell_text(cell) for cell in cells]
    
    # More specific header patterns that must match exactly
    header_patterns = [
        ["circuit/district", "title", "vacancy created by", "reason"],
        ["court", "vacancy created by", "reason", "vacancy date"],
        ["circuit", "title", "judge", "reason", "vacancy date"],
        ["circuit/district", "vacancy created by", "reason", "vacancy date"],  # 2015-01 pattern
    ]
    
    # Check for exact matches of header patterns
    for pattern in header_patterns:
        if len(cell_texts) >= len(pattern):
            # Check if the pattern matches at the start of the row
            if all(header == cell_texts[i].lower() 
                  for i, header in enumerate(pattern) if i < len(cell_texts)):
                return True
    
    # Handle title row pattern (e.g., "Judicial Emergencies" with a date)
    if len(cell_texts) >= 3 and \
       any(x in cell_texts[0].lower() for x in ['judicial emergencies', 'emergencies as of']):
        return True
    
    # Check for header-like structure with strong tags
    strong_cells = 0
    strong_texts = []
    
    for cell in cells:
        if hasattr(cell, 'find'):
            strong = cell.find('strong')
            if strong is not None and strong != -1:  # Check for both None and -1
                strong_cells += 1
                strong_texts.append(strong.get_text(strip=True).lower())
    
    # Require at least 3 strong cells with meaningful header text
    if strong_cells >= 3:
        # Count how many strong texts look like actual headers
        header_indicators = [
            'circuit', 'district', 'title', 'judge', 'vacancy', 
            'reason', 'date', 'days', 'pending', 'weighted', 'adjusted',
            'filing', 'judgeship', 'panel'  # Added more indicators
        ]
        header_like = sum(
            any(indicator in text for indicator in header_indicators)
            for text in strong_texts
        )
        # Require at least 3 strong texts that look like headers
        if header_like >= 3:
            return True
    
    return False


def _extract_legacy_format(table: Any) -> List[Dict[str, str]]:
    """Extract data from legacy format tables (pre-2016).
    
    These tables don't use thead/tbody and may have inconsistent formatting.
    """
    logger.debug("Starting extraction of legacy format table")
    records: List[Dict[str, str]] = []
    rows = table.find_all('tr', recursive=False)
    logger.debug(f"Found {len(rows)} total rows in table")
    
    def is_valid_header_row(cells: List[str]) -> bool:
        """Check if this is a valid header row with column labels."""
        if not cells or len(cells) < 5:  # Need enough columns to be a real header
            return False
            
        # Look for specific header patterns that indicate column labels
        header_indicators = [
            'circuit/district', 'title', 'vacancy created by', 'reason',
            'vacancy date', 'days pending', 'weighted', 'adjusted'
        ]
        
        # Check if any cell contains multiple header indicators (strong sign of column headers)
        cell_text = ' '.join(cell.lower() for cell in cells if cell)
        return sum(indicator in cell_text for indicator in header_indicators) >= 3
    
    # Find the header row (skip title rows and look for actual column headers)
    header_row_idx = None
    for i, row in enumerate(rows):
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        if _is_header_row(cells) and is_valid_header_row(cells):
            header_row_idx = i
            logger.debug(f"Found valid header row at index {i}: {cells}")
            break
    
    if header_row_idx is None:
        logger.warning("No valid header row found in legacy table")
        return []
    
    # Extract column indices from the header row
    header_row = rows[header_row_idx]
    column_indices = _get_column_indices(header_row)
    
    # Process data rows (skip rows before and including the header row)
    for row in rows[header_row_idx + 1:]:
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        
        # Skip empty rows or rows that look like headers/footers
        if not cells or _is_header_row(cells):
            continue
            
        record = {}
        for field, idx in column_indices.items():
            if idx < len(cells):
                record[field] = cells[idx]
        
        # Only add record if it has required fields
        if all(field in record for field in ['circuit_district', 'vacancy_date', 'days_pending']):
            records.append(record)
    
    logger.debug(f"Extracted {len(records)} records from legacy table")
    return records


def _detect_table_format(table: Any) -> str:
    """Detect the format of the table (modern or legacy).
    
    Modern tables (2016+) use <thead> and <tbody> sections with specific classes.
    Legacy tables (pre-2016) use simple table structure without these sections.
    
    Args:
        table: BeautifulSoup table element
        
    Returns:
        str: 'legacy' or 'modern' based on table structure
    """
    if not table:
        return 'modern'  # Default to modern for empty tables
    
    # Check for modern table structure (2020+)
    if table.find('thead') and table.find('tbody'):
        # Additional check for modern table classes to be sure
        if 'usa-table' in table.get('class', []) or 'views-table' in table.get('class', []):
            return 'modern'
    
    # Check for legacy table structure (pre-2020)
    # Look for specific patterns in the first few rows
    rows = table.find_all('tr', recursive=False, limit=5)  # Only check first few rows
    for row in rows:
        # Look for header-like content in cells
        cells = row.find_all(['td', 'th'])
        if not cells:
            continue
            
        # Check for common legacy header patterns
        text = ' '.join(cell.get_text(strip=True) for cell in cells).lower()
        if any(term in text for term in [
            'circuit/district', 'vacancy created by', 'vacancy date', 'days pending'
        ]):
            return 'legacy'
    
    # Default to modern if we can't determine
    return 'modern'


def extract_emergencies_table(html: str) -> List[Dict[str, str]]:
    """Extract judicial emergency declaration data from month-level HTML content."""
    logger.debug("Starting extraction of emergencies table")
    if not html or not isinstance(html, str):
        logger.warning("No HTML content provided")
        return []
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        
        if not table:
            logger.warning("No table found in HTML")
            return []
        
        # Determine table format and extract data accordingly
        table_format = _detect_table_format(table)
        logger.debug(f"Detected table format: {table_format}")
        
        if table_format == 'modern':
            records = _extract_modern_format(table)
        else:
            records = _extract_legacy_format(table)
        
        logger.info(f"Extracted total of {len(records)} emergency records")
        return records

    except Exception as e:
        # Log the error but don't fail the entire process
        logger.exception(f"Error extracting emergency data: {e}")
        return []
