"""
Data pipeline for scraping and processing judicial vacancy data.

This module provides functionality to parse and process judicial vacancy data from the US Courts website.
It includes functions for HTML parsing, data extraction, and file I/O operations.

This module can refer to config.py to determine which folder is being used as the raw data folder.

This module is NOT meant as a location for code for data transformations. Transformations should be done
in other modules such as features.py. This module shall deliver dataframes which represent their original
website sources (tables on HTML pages and PDFs) as accurately and unchanged as feasible, leaving the work
of data cleaning or feature creation to other code.
"""

import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Union

from bs4 import BeautifulSoup
import pandas as pd
import typer

from nomination_predictor.config import PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("data_pipeline.log")],
)
logger = logging.getLogger(__name__)  # noqa: F811

app = typer.Typer()


# Custom Exceptions
class DataPipelineError(Exception):
    """Base exception for data pipeline errors."""
    pass


class ParseError(Exception):
    """Base exception for parsing errors."""
    pass


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
        re.match(r'^[A-Za-z\s]+(?:Circuit|Court|District)', court, re.IGNORECASE)  # 1st Circuit
    )


def _extract_modern_format(table: Any) -> List[Dict[str, str]]:
    """Extract data from modern (2016+) HTML format.
    
    Args:
        table: BeautifulSoup table element
        
    Returns:
        List of dictionaries with extracted data
    """
    # Extract headers from thead if available
    headers = []
    thead = table.find('thead')
    if thead:
        header_row = thead.find('tr')
        if header_row:
            headers = [
                th.get_text(strip=True).lower().replace(' ', '_')
                for th in header_row.find_all(['th', 'td'])
            ]
    
    # If no headers in thead, try to find them in the first row of tbody
    if not headers:
        tbody = table.find('tbody')
        if tbody:
            first_row = tbody.find('tr')
            if first_row:
                headers = [
                    th.get_text(strip=True).lower().replace(' ', '_')
                    for th in first_row.find_all(['th', 'td'])
                ]
    
    return _extract_table_data(table, headers)


def _extract_legacy_format(table: Any) -> List[Dict[str, str]]:
    """Extract data from legacy format tables (pre-2016).
    
    These tables don't use thead/tbody and have header rows mixed in with data.
    """
    records = []
    rows = table.find_all('tr')
    
    for row in rows:
        cells = row.find_all(['th', 'td'])
        if not cells:
            continue
            
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        
        # Skip header rows and empty rows
        if not cell_texts or not cell_texts[0]:
            continue
            
        # Use the reusable court validation function
        if not is_valid_court_identifier(cell_texts[0]):
            continue
            
        # Rest of the extraction logic remains the same
        if len(cell_texts) >= 4:  # Minimum cells we expect
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
                    
            records.append(record)
    
    return records


def _extract_table_data(table: Any, headers: List[str], skip_rows: int = 0) -> List[Dict[str, str]]:
    """Extract data from a table with the given headers.
    
    Args:
        table: BeautifulSoup table element
        headers: List of header names
        skip_rows: Number of rows to skip (e.g., for header rows)
        
    Returns:
        List of dictionaries with extracted data
    """
    rows = []
    tbody = table.find('tbody') or table  # Use tbody if it exists, otherwise use the whole table
    
    for i, row in enumerate(tbody.find_all('tr')):
        if i < skip_rows:
            continue
            
        cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
        if not cells:
            continue
            
        # Create a dictionary with the row data
        row_data = {}
        for j, cell in enumerate(cells):
            if j < len(headers):
                row_data[headers[j]] = cell
            else:
                row_data[f'column_{j}'] = cell
                
        if row_data:  # Only add non-empty rows
            rows.append(row_data)
    
    return rows


def _detect_table_format(table: Any) -> str:
    """Detect the format of the table (modern or legacy).
    
    Modern tables typically have <thead> and <tbody> sections, while legacy tables
    have header rows with <strong> tags or specific text patterns.
    
    Args:
        table: BeautifulSoup table element
        
    Returns:
        'modern' or 'legacy' based on the table structure
    """
    # Check for modern format indicators
    if table.find('thead') or table.find('tbody'):
        return 'modern'
    
    # Check for legacy format indicators by examining the first few rows
    rows = table.find_all('tr', limit=5)  # Only check first 5 rows for efficiency
    
    # Common header patterns in legacy tables
    legacy_header_indicators = [
        'circuit', 'district', 'incumbent', 'vacancy', 'reason', 
        'nominee', 'date', 'judge', 'court'
    ]
    
    for row in rows:
        # Check for header cells with <strong> tags
        strong_cells = row.find_all(['th', 'td'])
        strong_texts = [cell.get_text(strip=True).lower() for cell in strong_cells]
        
        # Count how many cells contain legacy header indicators
        matching_cells = sum(
            1 for text in strong_texts 
            if any(indicator in text for indicator in legacy_header_indicators)
        )
        
        # If we find a row where most cells look like headers, it's likely a legacy table
        if matching_cells >= 2 and len(strong_texts) > 0 and matching_cells / len(strong_texts) >= 0.5:
            return 'legacy'
            
        # Also check for text directly in cells (without strong tags)
        cell_texts = [cell.get_text(strip=True).lower() for cell in row.find_all(['th', 'td'])]
        if any(indicator in cell for cell in cell_texts[:4] for indicator in legacy_header_indicators):
            return 'legacy'
    
    # Default to modern if we can't determine with high confidence
    return 'modern'


def extract_vacancy_table(html: str) -> List[Dict[str, Any]]:
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
    try:
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        if not table:
            return []
        
        # Detect table format and extract data accordingly
        format_type = _detect_table_format(table)
        if format_type == 'modern':
            records = _extract_modern_format(table)
        else:
            records = _extract_legacy_format(table)
        
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
        for record in records:
            standardized = {}
            for old_key, value in record.items():
                new_key = field_mapping.get(old_key.lower(), old_key)
                standardized[new_key] = value.strip() if isinstance(value, str) else value
            standardized_records.append(standardized)
        
        return standardized_records

    except Exception as e:
        raise ParseError(f"Error parsing HTML table: {e}") from e


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of records to a pandas DataFrame.

    Args:
        records: List of dictionaries containing the data

    Returns:
        DataFrame containing the processed data
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Convert numeric columns
    if "seat" in df.columns:
        df["seat"] = pd.to_numeric(df["seat"], errors="coerce")

    # Convert date columns
    date_columns = ["vacancy_date", "nomination_date"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def save_to_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df: DataFrame to save
        path: Output file path

    Raises:
        DataPipelineError: If saving fails
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Successfully saved data to {path}")
    except Exception as e:
        raise DataPipelineError(f"Failed to save CSV to {path}: {e}")


@app.command()
def main(output_dir: Path = PROCESSED_DATA_DIR, output_filename: str = "judicial_vacancies.csv"):
    """
    Main entry point for the data pipeline.

    Args:
        output_dir: Directory to save output files
        output_filename: Name of the output CSV file
    """
    from nomination_predictor.web_utils import fetch_html, generate_or_fetch_archive_urls

    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        logger.info("Starting data pipeline...")

        # Fetch and process data
        urls = generate_or_fetch_archive_urls()
        all_records = []

        for url in urls:
            try:
                html = fetch_html(url)
                records = extract_vacancy_table(html)
                all_records.extend(records)
                logger.debug(f"Processed {len(records)} records from {url}")
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                continue

        # Convert to DataFrame and save
        if all_records:
            df = records_to_dataframe(all_records)
            save_to_csv(df, output_path)
            logger.success(f"Successfully processed {len(df)} records to {output_path}")
        else:
            logger.warning("No records were processed")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    app()
