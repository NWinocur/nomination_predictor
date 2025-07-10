"""
Data pipeline for scraping and processing judicial vacancy data.

This module provides functionality to parse and process judicial vacancy data from the US Courts website.
It coordinates between different specialized scrapers for vacancies, emergencies, and confirmations.

This module is responsible for the initial data extraction and saving raw parsed data to the raw data directory.
For data cleaning and transformation, see the data_cleaning.py and features.py modules.
"""

from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Union

import pandas as pd
import typer

from nomination_predictor.config import RAW_DATA_DIR
from nomination_predictor.confirmations_scraper import extract_confirmations_table
from nomination_predictor.emergencies_scraper import extract_emergencies_table
from nomination_predictor.vacancy_scraper import extract_vacancy_table
from nomination_predictor.web_utils import fetch_html, generate_month_links

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("data_pipeline.log")],
)
logger = logging.getLogger(__name__)

app = typer.Typer()


class DataPipelineError(Exception):
    """Base exception for data pipeline errors."""
    pass


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


def save_dataframe_to_csv(df: pd.DataFrame, filename_without_extension: str, output_dir: Path) -> None:
    """
    Save a DataFrame to a CSV file with error handling.
    
    Args:
        df: DataFrame to save
        filename_without_extension: Output filename (without .csv extension)
        output_dir: Directory to save the file in
    """
    try:
        output_path = output_dir / f"{filename_without_extension}.csv"
        df.to_csv(output_path, index=False, sep='|')
        logger.info(f"Successfully saved {len(df)} records to {output_path}")
    except Exception as e:
        logger.error(f"Error saving {filename_without_extension}: {e}")
        raise


def parse_circuit_court(court_str: str) -> tuple[int | None, str]:
    """
    Parse a circuit/court identifier string into its components.
    
    Args:
        court_str: The court string to parse (e.g., "09 - CA-N", "DC - DC", "IT", "IT -")
        
    Returns:
        A tuple of (circuit, court) where:
        - circuit: The circuit number as an int, or None if not applicable
        - court: The court/district identifier as a string
        
    Raises:
        ValueError: If the input string cannot be parsed
    """
    if not court_str or not isinstance(court_str, str):
        raise ValueError(f"Invalid court string: {court_str}")
    
    # Clean up the input string
    court_str = court_str.strip()
    
    # Special cases for courts without circuits
    special_courts: set[str] = {"IT", "CL", "SC", "FED", "DC"}
    
    # Check for special court code with trailing dash (e.g., "IT -")
    for code in special_courts:
        if court_str.startswith(f"{code} -"):
            return None, code
    
    if court_str in special_courts:
        return None, court_str
    
    # Handle the standard format: "NN - XX-YY" or "NN - XX"
    parts: list[str] = court_str.split(' - ', 1)
    if len(parts) != 2:
        # If we can't split on ' - ', check if it's a special court
        if court_str in special_courts:
            return None, court_str
        raise ValueError(f"Could not parse court string: {court_str}")
    
    circuit_part: str
    court_part: str
    circuit_part, court_part = (part.strip() for part in parts)
    
    # If the court part is empty after stripping, check if the circuit part is a special court
    if not court_part and circuit_part in special_courts:
        return None, circuit_part
    
    # Parse circuit number
    try:
        # Map special circuit codes to numeric values we defined in project's data dictionary
        CIRCUIT_CODE_MAP: dict[str, int] = {
            "DC": 12,   # DC Circuit
            "FD": 13,   # Federal Circuit
            "FED": 13,  # Federal Circuit
            "IT": 13,   # International Trade Circuit
            "CL": 13,   # Federal Claims Circuit
            "SC": 13,   # Supreme Court Circuit
            # can add other special codes as needed
        }
        
        # Check if circuit_part is a special code
        if circuit_part in CIRCUIT_CODE_MAP:
            circuit = CIRCUIT_CODE_MAP[circuit_part]
        else:
            circuit = int(circuit_part)
        
        # Validate circuit number (1-11 for normal circuits, 12 for DC, 13 for FED/other special)
        if not (1 <= circuit <= 13):
            raise ValueError(f"Invalid circuit number: {circuit}")
            
    except (ValueError, TypeError) as e:
        # If we can't parse as a circuit number, check if it's a special court code
        if court_part in special_courts:
            return None, court_part
        if circuit_part in special_courts:
            return None, circuit_part
        raise ValueError(f"Invalid circuit number format: {circuit_part}") from e
    
    return circuit, court_part


def fetch_data(
    page_type: str, 
    current_year: int = datetime.now().year,
    years_back: int = 15
) -> pd.DataFrame:
    """
    Fetch and process data for a specific page type (vacancies, confirmations, emergencies).
    
    Args:
        page_type: Type of data to fetch ('vacancies', 'confirmations', or 'emergencies')
        current_year: Current year for generating date ranges
        years_back: Number of years back to fetch data for
        
    Returns:
        DataFrame containing the processed data
    """
    all_records = []
    start_year = current_year - years_back
    
    logger.info(f"Fetching {page_type} data from {start_year} to {current_year}")
    
    # Generate month links for each year and process each month
    for year in range(start_year, current_year + 1):
        try:
            month_links = generate_month_links(year, page_type=page_type)
            logger.info(f"Processing {len(month_links)} months for {page_type} in {year}")
            
            for link in month_links:
                try:
                    # Skip future months in the current year
                    if year == current_year and int(link['month']) > datetime.now().month:
                        continue
                        
                    full_url = f"https://www.uscourts.gov{link['url']}"
                    logger.debug(f"Fetching {full_url}")
                    html = fetch_html(full_url)
                    
                    # Use the appropriate extractor based on page type
                    if page_type == 'vacancies':
                        records = extract_vacancy_table(html)
                    elif page_type == 'confirmations':
                        records = extract_confirmations_table(html)
                    elif page_type == 'emergencies':
                        records = extract_emergencies_table(html)
                    else:
                        logger.error(f"Invalid page type: {page_type}")
                        continue
                        
                    if records:
                        # Add year and month to each record for tracking when the report came from
                        for record in records:
                            record.update({
                                'source_year': str(year),
                                'source_month': link['month'],  # Already a string
                                'source_page_type': page_type
                            })
                        all_records.extend(records)
                        logger.info(f"Processed {len(records)} {page_type} records from {year}-{link['month']}")
                        
                except Exception as e:
                    logger.error(f"Error processing {link['url']}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing year {year} for {page_type}: {e}")
            continue
    
    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


@app.command()
def main(
    output_dir: Path = RAW_DATA_DIR,
    output_filename: str = "judicial_data.csv",
    years_back: int = typer.Option(
        15,
        help="Number of years of historical data to fetch"
    )
):
    """
    Main entry point for the data pipeline.

    This script scrapes and processes judicial vacancy data from the US Courts website,
    saving the raw parsed data to the raw data directory.

    Args:
        output_dir: Directory to save output files (default: RAW_DATA_DIR)
        output_filename: Name of the output CSV file
        years_back: Number of years of historical data to fetch
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        current_year = datetime.now().year

        logger.info(f"Starting data pipeline. Output will be saved to: {output_path}")
        
        # Process vacancies and confirmations
        dfs = []
        
        # Fetch and process vacancies
        logger.info("Processing vacancy data...")
        vacancies_df = fetch_data('vacancies', current_year, years_back)
        if not vacancies_df.empty:
            dfs.append(vacancies_df)
            logger.info(f"Processed {len(vacancies_df)} vacancy records")
            
        # Fetch and process confirmations
        logger.info("Processing confirmation data...")
        confirmations_df = fetch_data('confirmations', current_year, years_back)
        if not confirmations_df.empty:
            dfs.append(confirmations_df)
            logger.info(f"Processed {len(confirmations_df)} confirmation records")
            
        # Fetch and process emergencies
        logger.info("Processing judicial emergency data...")
        emergencies_df = fetch_data('emergencies', current_year, years_back)
        if not emergencies_df.empty:
            dfs.append(emergencies_df)
            logger.info(f"Processed {len(emergencies_df)} emergency records")
        
        # Combine all data
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Save the combined data
            save_dataframe_to_csv(combined_df, output_path.stem, output_dir)
            
            # Also save individual datasets for reference
            if not vacancies_df.empty:
                save_dataframe_to_csv(vacancies_df, "judicial_vacancies", output_dir)
            if not confirmations_df.empty:
                save_dataframe_to_csv(confirmations_df, "judicial_confirmations", output_dir)
            if not emergencies_df.empty:
                save_dataframe_to_csv(emergencies_df, "judicial_emergencies", output_dir)
                
            return combined_df
        else:
            logger.warning("No records were processed")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    app()
