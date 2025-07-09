"""
Vacancy table extraction for judicial nominations data.

This module handles the extraction of vacancy data from HTML tables in the US Courts website.
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


def _extract_modern_format(table: Any) -> List[Dict[str, str]]:
    """Extract data from modern (2016+) HTML format.
    
    Modern tables use thead/tbody structure and have consistent column ordering.
    """
    records: List[Dict[str, str]] = []
    
    # Find all rows in the tbody (skip thead)
    tbody = table.find('tbody')
    rows = tbody.find_all('tr') if tbody else table.find_all('tr')
    
    for row in rows:
        cells = row.find_all('td')
        if not cells:
            continue
            
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        
        # Skip rows that don't have a valid court identifier in the first cell
        if not cell_texts or not is_valid_court_identifier(cell_texts[0]):
            continue
            
        record = {
            'court': cell_texts[0],
            'incumbent': cell_texts[1] if len(cell_texts) > 1 else '',
            'vacancy_reason': cell_texts[2] if len(cell_texts) > 2 else '',
            'vacancy_date': cell_texts[3] if len(cell_texts) > 3 else '',
        }
        
        # Add nominee information if available
        if len(cell_texts) > 4 and cell_texts[4]:
            record['nominee'] = cell_texts[4]
            if len(cell_texts) > 5 and cell_texts[5]:
                record['nomination_date'] = cell_texts[5]
            if len(cell_texts) > 6 and cell_texts[6]:
                record['confirmation_date'] = cell_texts[6]
                
        records.append(record)
    
    return records


def _extract_legacy_format(table: Any) -> List[Dict[str, str]]:
    """Extract data from legacy format tables (pre-2016).
    
    These tables don't use thead/tbody and have header rows mixed in with data.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.debug("Starting extraction of legacy format table")
    
    records: List[Dict[str, str]] = []
    rows = table.find_all('tr')
    logger.debug(f"Found {len(rows)} rows in table")
    
    for i, row in enumerate(rows, 1):
        cells = row.find_all(['th', 'td'])
        if not cells:
            logger.debug(f"Row {i}: No cells found, skipping")
            continue
            
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        logger.debug(f"Row {i} cell texts: {cell_texts}")
        
        # Skip header rows and empty rows
        if not cell_texts or not cell_texts[0]:
            logger.debug(f"Row {i}: Empty first cell, skipping")
            continue
            
        # Use the reusable court validation function
        if not is_valid_court_identifier(cell_texts[0]):
            logger.debug(f"Row {i}: Invalid court identifier: {cell_texts[0]}")
            continue
            
        # Minimum cells we expect
        if len(cell_texts) >= 4:
            record = {
                'court': cell_texts[0],
                'incumbent': cell_texts[1],
                'vacancy_reason': cell_texts[2],
                'vacancy_date': cell_texts[3],
            }
            
            # Add nominee and dates if available
            if len(cell_texts) > 4 and cell_texts[4]:  # Nominee
                record['nominee'] = cell_texts[4]
                if len(cell_texts) > 5 and cell_texts[5]:  # Nomination date
                    record['nomination_date'] = cell_texts[5]
                if len(cell_texts) > 6 and cell_texts[6]:  # Confirmation date
                    record['confirmation_date'] = cell_texts[6]
            
            logger.debug(f"Row {i}: Extracted record: {record}")
            records.append(record)
        else:
            logger.debug(f"Row {i}: Insufficient cells ({len(cell_texts)} < 4), skipping")
    
    logger.info(f"Extracted {len(records)} records from legacy table")
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
    import logging
    logger = logging.getLogger(__name__)
    
    # Check for modern table structure
    has_thead = table.find('thead') is not None
    has_tbody = table.find('tbody') is not None
    
    logger.debug(f"Table structure - thead: {has_thead}, tbody: {has_tbody}")
    
    if has_thead and has_tbody:
        logger.debug("Detected modern table format (has both thead and tbody)")
        return 'modern'
    
    # Check for legacy table indicators
    rows = table.find_all('tr', limit=5)  # Only check first few rows
    logger.debug(f"Checking first {len(rows)} rows for legacy format indicators")
    
    # Common header patterns in legacy tables
    legacy_header_indicators = [
        'circuit', 'district', 'incumbent', 'vacancy', 'reason', 
        'nominee', 'date', 'judge', 'court'
    ]
    
    for i, row in enumerate(rows, 1):
        # Check for header cells with specific text
        header_cells = row.find_all(['th', 'td'])
        if not header_cells:
            logger.debug(f"Row {i}: No header cells found")
            continue
            
        # Get text from all cells in the row
        cell_texts = [cell.get_text(strip=True).lower() for cell in header_cells]
        logger.debug(f"Row {i} cell texts: {cell_texts}")
        
        # Count how many cells contain legacy header indicators
        matching_cells = sum(
            1 for text in cell_texts 
            if any(indicator in text for indicator in legacy_header_indicators)
        )
        
        # If we find a row where most cells look like headers, it's likely a legacy table
        if matching_cells >= 2 and len(cell_texts) > 0 and matching_cells / len(cell_texts) >= 0.5:
            logger.debug(f"Row {i}: Detected legacy format based on header patterns")
            return 'legacy'
            
        # Also check for bold text in first cell (common in legacy headers)
        first_cell = header_cells[0]
        if first_cell.find(['strong', 'b']):
            logger.debug(f"Row {i}: Detected legacy format based on bold text in first cell")
            return 'legacy'
    
    # Default to modern if no legacy indicators found
    logger.debug("No legacy format indicators found, defaulting to modern format")
    return 'modern'


def extract_vacancy_table(html: str) -> List[Dict[str, str]]:
    """
    Extract judicial vacancy data from month-level HTML content.

    Handles both older (pre-2016) and newer (2016+) HTML formats.

    Args:
        html: HTML content containing a table with judicial vacancy data

    Returns:
        List of dictionaries containing extracted data with standardized fields:
        - court: Name of the court/district (required)
        - vacancy_date: Date the vacancy occurred (required)
        - incumbent: Name of the outgoing judge (required)
        - vacancy_reason: Reason for the vacancy (required)
        - nominee: Name of the nominee (if any)
        - nomination_date: Date of nomination (if any)
        - confirmation_date: Date of confirmation (if any)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if not table:
        logger.debug("No table found in HTML content")
        return []
    
    # Log first 500 chars of table HTML for debugging
    table_html = str(table)[:500] + '...' if len(str(table)) > 500 else str(table)
    logger.debug(f"Found table. First 500 chars: {table_html}")
    
    # Detect table format and extract data accordingly
    format_type = _detect_table_format(table)
    logger.info(f"Detected table format: {format_type}")
    
    if format_type == 'modern':
        logger.debug("Extracting modern format table")
        records = _extract_modern_format(table)
    else:
        logger.debug("Extracting legacy format table")
        records = _extract_legacy_format(table)
    
    logger.info(f"Extracted {len(records)} raw records")
    
    # Standardize field names
    field_mapping = {
        # Modern format field names
        'vacant_judgeship': 'incumbent',
        'date_of_vacancy': 'vacancy_date',
        'reason_for_vacancy': 'vacancy_reason',
        # Legacy format field names
        'judge': 'incumbent',
        'date': 'vacancy_date',
        'reason': 'vacancy_reason',
        'date_of_nomination': 'nomination_date',
        'date_of_confirmation': 'confirmation_date',
    }
    
    standardized_records = []
    for i, record in enumerate(records, 1):
        try:
            standardized = {}
            for old_key, value in record.items():
                new_key = field_mapping.get(old_key.lower(), old_key)
                standardized[new_key] = value.strip() if isinstance(value, str) else value
            standardized_records.append(standardized)
            logger.debug(f"Processed record {i}: {standardized}")
        except Exception as e:
            logger.warning(f"Error processing record {i}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(standardized_records)} records")
    return standardized_records
