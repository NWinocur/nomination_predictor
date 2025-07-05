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


def extract_vacancy_table(html: str) -> List[Dict[str, Any]]:
    """
    Extract judicial vacancy data from month-level HTML content.

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
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return []

        # Get headers and normalize them (lowercase, replace spaces with underscores)
        headers = []
        header_row = table.find("tr")
        if header_row:
            headers = [
                th.get_text(strip=True).lower().replace(" ", "_")
                for th in header_row.find_all(["th", "td"])
            ]
        else:
            # If no header row, use default column names based on position
            headers = [
                "court", 
                "vacancy_date", 
                "incumbent", 
                "vacancy_reason", 
                "nominee", 
                "nomination_date", 
                "confirmation_date"
            ]

        rows = []
        # Skip header row if it exists
        for row in table.find_all("tr")[1 if header_row else 0:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if not cells:  # Skip empty rows
                continue

            # Create a dictionary with the row data
            row_data = {}
            for i, cell in enumerate(cells):
                if i < len(headers):
                    row_data[headers[i]] = cell
                else:
                    # If we have more cells than headers, add them with generic keys
                    row_data[f"column_{i}"] = cell
            
            if row_data:  # Only add non-empty rows
                rows.append(row_data)

        return rows

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
