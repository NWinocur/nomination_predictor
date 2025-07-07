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


def save_to_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Save a DataFrame to a CSV file using pipe (|) as the delimiter.
    
    We use pipe as the delimiter because it's much less likely to appear in our data
    than commas, dashes, or slashes, which are common in court names and dates.

    Args:
        df: DataFrame to save
        path: Output file path

    Raises:
        DataPipelineError: If saving fails
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, sep='|')
        logger.info(f"Successfully saved data to {path} using pipe (|) delimiter")
    except Exception as e:
        raise DataPipelineError(f"Failed to save CSV to {path}: {e}")


def fetch_and_process_data(
    page_type: str, 
    output_dir: Path, 
    current_year: int,
    years_back: int = 15
) -> pd.DataFrame:
    """
    Fetch and process data for a specific page type (vacancies, confirmations, emergencies).
    
    Args:
        page_type: Type of data to fetch ('vacancies', 'confirmations', or 'emergencies')
        output_dir: Directory to save output files
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
                    else:  # emergencies
                        # TODO: Add emergencies scraper when implemented
                        continue
                        
                    if records:
                        # Add year and month to each record for tracking
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
        vacancies_df = fetch_and_process_data('vacancies', output_dir, current_year, years_back)
        if not vacancies_df.empty:
            dfs.append(vacancies_df)
            logger.info(f"Processed {len(vacancies_df)} vacancy records")
            
        # Fetch and process confirmations
        logger.info("Processing confirmation data...")
        confirmations_df = fetch_and_process_data('confirmations', output_dir, current_year, years_back)
        if not confirmations_df.empty:
            dfs.append(confirmations_df)
            logger.info(f"Processed {len(confirmations_df)} confirmation records")
            
        # TODO: Uncomment when emergencies scraper is implemented
        # emergencies_df = fetch_and_process_data('emergencies', output_dir, current_year, years_back)
        # if not emergencies_df.empty:
        #     dfs.append(emergencies_df)
        
        # Combine all data
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Save the combined data
            save_to_csv(combined_df, output_path)
            logger.info(f"Successfully saved {len(combined_df)} total records to {output_path}")
            
            # Also save individual datasets for reference
            if not vacancies_df.empty:
                save_to_csv(vacancies_df, output_dir / "judicial_vacancies.csv")
            if not confirmations_df.empty:
                save_to_csv(confirmations_df, output_dir / "judicial_confirmations.csv")
                
            return combined_df
        else:
            logger.warning("No records were processed")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    app()
