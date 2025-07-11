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
from typing import Any, Dict, List, Optional, TypedDict, Union

from loguru import logger
import numpy as np
import pandas as pd
import typer

from nomination_predictor.config import RAW_DATA_DIR
from nomination_predictor.confirmations_scraper import extract_confirmations_table
from nomination_predictor.congress_api import CongressAPIClient
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


def save_dataframe_to_csv(
    df: pd.DataFrame, filename_without_extension: str, output_dir: Path
) -> None:
    """
    Save a DataFrame to a CSV file with error handling.

    Args:
        df: DataFrame to save
        filename_without_extension: Output filename (without .csv extension)
        output_dir: Directory to save the file in
    """
    try:
        output_path = output_dir / f"{filename_without_extension}.csv"
        df.to_csv(output_path, index=False, sep="|")
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
    parts: list[str] = court_str.split(" - ", 1)
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
            "DC": 12,  # DC Circuit
            "FD": 13,  # Federal Circuit
            "FED": 13,  # Federal Circuit
            "IT": 13,  # International Trade Circuit
            "CL": 13,  # Federal Claims Circuit
            "SC": 13,  # Supreme Court Circuit
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
    page_type: str, current_year: int = datetime.now().year, years_back: int = 15
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
                    if year == current_year and int(link["month"]) > datetime.now().month:
                        continue

                    full_url = f"https://www.uscourts.gov{link['url']}"
                    logger.debug(f"Fetching {full_url}")
                    html = fetch_html(full_url)

                    # Use the appropriate extractor based on page type
                    if page_type == "vacancies":
                        records = extract_vacancy_table(html)
                    elif page_type == "confirmations":
                        records = extract_confirmations_table(html)
                    elif page_type == "emergencies":
                        records = extract_emergencies_table(html)
                    else:
                        logger.error(f"Invalid page type: {page_type}")
                        continue

                    if records:
                        # Add year and month to each record for tracking when the report came from
                        for record in records:
                            record.update(
                                {
                                    "source_year": str(year),
                                    "source_month": link["month"],  # Already a string
                                    "source_page_type": page_type,
                                }
                            )
                        all_records.extend(records)
                        logger.info(
                            f"Processed {len(records)} {page_type} records from {year}-{link['month']}"
                        )

                except Exception as e:
                    logger.error(f"Error processing {link['url']}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing year {year} for {page_type}: {e}")
            continue

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def fetch_data_from_congress_api(
    current_congress: int = 118, congresses_back: int = 5
) -> pd.DataFrame:
    """
    Fetch judicial nomination data from Congress.gov API.

    Args:
        current_congress: The current congress number to start fetching from
        congresses_back: Number of congresses to go back in history

    Returns:
        DataFrame containing judicial nomination data
    """
    try:
        client = CongressAPIClient()
    except ValueError as e:
        logger.error(f"Failed to initialize Congress.gov API client: {e}")
        logger.warning("To use Congress.gov API, set CONGRESS_API_KEY environment variable")
        return pd.DataFrame()

    all_records = []

    # Iterate through congresses
    for congress in range(current_congress, current_congress - congresses_back - 1, -1):
        try:
            logger.info(f"Fetching judicial nominations for {congress}th Congress")
            congress_records = client.get_judicial_nominations(congress, congresses_back = congresses_back)
            logger.info(f"Retrieved {len(congress_records)} judicial nomination records")
            all_records.extend(congress_records)

        except Exception as e:
            logger.error(f"Error fetching data for {congress}th Congress: {e}")

    if not all_records:
        logger.warning("No judicial nomination records found")
        return pd.DataFrame()

    logger.success(
        f"Successfully retrieved {len(all_records)} judicial nomination records across {congresses_back + 1} congresses"
    )
    return records_to_dataframe(all_records)


class SchemaComparison(TypedDict):
    common_fields: List[str]
    missing_from_api: List[str]
    extra_in_api: List[str]
    coverage_percentage: float

class DataQuality(TypedDict):
    null_percentage: Dict[str, float]
    high_null_fields: List[str]

class SchemaValidationResults(TypedDict):
    api_record_count: int
    vacancy_record_count: Optional[int]
    confirmation_record_count: Optional[int]
    schema_comparison: Dict[str, Dict[str, Any]]  # Nested structure for vacancy/confirmation
    data_quality: DataQuality
    compatibility_score: float
    missing_critical_vacancy_fields: Optional[List[str]]
    missing_critical_confirmation_fields: Optional[List[str]]

def compare_and_validate_api_data(
    api_df: pd.DataFrame,
    legacy_vacancy_path: Path = RAW_DATA_DIR / "judicial_data.csv",
    legacy_confirmation_path: Path = RAW_DATA_DIR / "judicial_confirmations.csv",
) -> SchemaValidationResults:
    """
    Compare and validate Congress.gov API-derived data against legacy data schemas.

    This function analyzes the schema compatibility between API-derived nomination data
    and the legacy scraped data, identifying matching and missing fields to ensure
    the new data source can be used alongside or in place of the legacy pipeline.

    Args:
        api_df: DataFrame containing Congress.gov API-derived data
        legacy_vacancy_path: Path to the legacy vacancy data CSV
        legacy_confirmation_path: Path to the legacy confirmation data CSV

    Returns:
        Dictionary containing validation results
    """
    results: SchemaValidationResults = {
        "api_record_count": len(api_df),
        "vacancy_record_count": None,
        "confirmation_record_count": None,
        "schema_comparison": {
            "vacancy": {
                "common_fields": [],
                "missing_from_api": [],
                "extra_in_api": [],
                "coverage_percentage": 0.0,
            },
            "confirmation": {
                "common_fields": [],
                "missing_from_api": [],
                "extra_in_api": [],
                "coverage_percentage": 0.0,
            }
        },
        "data_quality": {
            "null_percentage": {},
            "high_null_fields": [],
        },
        "compatibility_score": 0.0,
        "missing_critical_vacancy_fields": None,
        "missing_critical_confirmation_fields": None,
    }

    # Load legacy data if files exist
    legacy_vacancy_df = pd.DataFrame()
    legacy_confirmation_df = pd.DataFrame()

    if legacy_vacancy_path.exists():
        try:
            legacy_vacancy_df = pd.read_csv(legacy_vacancy_path, sep="|")
            results["vacancy_record_count"] = len(legacy_vacancy_df)
            logger.info(f"Loaded {len(legacy_vacancy_df)} legacy vacancy records")
        except Exception as e:
            logger.error(f"Error loading legacy vacancy data: {e}")

    if legacy_confirmation_path.exists():
        try:
            legacy_confirmation_df = pd.read_csv(legacy_confirmation_path, sep="|")
            results["confirmation_record_count"] = len(legacy_confirmation_df)
            logger.info(f"Loaded {len(legacy_confirmation_df)} legacy confirmation records")
        except Exception as e:
            logger.error(f"Error loading legacy confirmation data: {e}")

    # Compare schemas
    api_columns = set(api_df.columns)

    # Vacancy data schema comparison
    if not legacy_vacancy_df.empty:
        vacancy_columns = set(legacy_vacancy_df.columns)
        common_vacancy_fields = api_columns.intersection(vacancy_columns)
        missing_from_api = vacancy_columns - api_columns
        extra_in_api = api_columns - vacancy_columns

        results["schema_comparison"]["vacancy"] = {
            "common_fields": list(common_vacancy_fields),
            "missing_from_api": list(missing_from_api),
            "extra_in_api": list(extra_in_api),
            "coverage_percentage": round(
                len(common_vacancy_fields) / len(vacancy_columns) * 100, 2
            ),
        }

        # Check critical fields
        critical_vacancy_fields = [
            "circuit_district",
            "court",
            "circuit",
            "vacancy_date",
            "nominee",
        ]
        missing_critical = [
            field for field in critical_vacancy_fields if field in missing_from_api
        ]
        results["schema_comparison"]["missing_critical_vacancy_fields"] = missing_critical

    # Confirmation data schema comparison
    if not legacy_confirmation_df.empty:
        confirmation_columns = set(legacy_confirmation_df.columns)
        common_confirmation_fields = api_columns.intersection(confirmation_columns)
        missing_from_api_conf = confirmation_columns - api_columns
        extra_in_api_conf = api_columns - confirmation_columns

        results["schema_comparison"]["confirmation"] = {
            "common_fields": list(common_confirmation_fields),
            "missing_from_api": list(missing_from_api_conf),
            "extra_in_api": list(extra_in_api_conf),
            "coverage_percentage": round(
                len(common_confirmation_fields) / len(confirmation_columns) * 100, 2
            ),
        }

        # Check critical fields
        critical_confirmation_fields = ["nominee", "court", "confirmation_date"]
        missing_critical_conf = [
            field for field in critical_confirmation_fields if field in missing_from_api_conf
        ]
        results["schema_comparison"]["missing_critical_confirmation_fields"] = (
            missing_critical_conf
        )

    # Data quality checks
    if not api_df.empty:
        null_percentage = {}
        for column in api_df.columns:
            null_pct = (api_df[column].isna().sum() / len(api_df)) * 100
            null_percentage[column] = round(null_pct, 2)

        results["data_quality"] = {
            "null_percentage": null_percentage,
            "high_null_fields": [col for col, pct in null_percentage.items() if pct > 50],
        }

    # Calculate overall compatibility score
    compatibility_scores = []

    if "vacancy" in results["schema_comparison"]:
        vacancy_score = results["schema_comparison"]["vacancy"]["coverage_percentage"] / 100
        critical_penalty = (
            len(results["schema_comparison"]["missing_critical_vacancy_fields"]) * 0.2
        )
        vacancy_final_score = max(0, vacancy_score - critical_penalty)
        compatibility_scores.append(vacancy_final_score)

    if "confirmation" in results["schema_comparison"]:
        confirmation_score = (
            results["schema_comparison"]["confirmation"]["coverage_percentage"] / 100
        )
        critical_penalty = (
            len(results["schema_comparison"]["missing_critical_confirmation_fields"]) * 0.2
        )
        confirmation_final_score = max(0, confirmation_score - critical_penalty)
        compatibility_scores.append(confirmation_final_score)

    if compatibility_scores:
        results["compatibility_score"] = round(np.mean(compatibility_scores) * 100, 2)

    return results


@app.command()
def main(
    output_dir: Path = RAW_DATA_DIR,
    output_filename: str = "judicial_data.csv",
    years_back: int = typer.Option(15, help="Number of years of historical data to fetch"),
    use_congress_api: bool = typer.Option(
        False, help="Use Congress.gov API instead of web scraping"
    ),
    current_congress: int = typer.Option(
        118, help="Current congress number (only used with Congress.gov API)"
    ),
):
    """
    Main entry point for the data pipeline.

    This script scrapes and processes judicial vacancy data from the US Courts website,
    saving the raw parsed data to the raw data directory.

    Args:
        output_dir: Directory to save output files (default: RAW_DATA_DIR)
        output_filename: Name of the output CSV file
        years_back: Number of years of historical data to fetch
        use_congress_api: Whether to use the Congress.gov API instead of web scraping
        current_congress: Current congress number (only used with Congress.gov API)
    """
    # [existing setup code]

    if use_congress_api:
        logger.info("Using Congress.gov API for data collection")

        # Calculate congresses back based on years back (approx. 2 years per congress)
        congresses_back = (years_back + 1) // 2

        # Fetch data using Congress API
        try:
            # For now, just output to a separate file to avoid conflicts
            api_df = fetch_data_from_congress_api(
                current_congress=current_congress, congresses_back=congresses_back
            )

            if not api_df.empty:
                api_output_path = output_dir / "congress_api_judicial_data.csv"
                save_dataframe_to_csv(api_df, "congress_api_judicial_data", output_dir)
                logger.success(f"Congress.gov API data saved to: {api_output_path}")

                # Validate API data against legacy schemas
                validation_results = compare_and_validate_api_data(api_df)
                logger.info("API data validation results:")
                for key, value in validation_results.items():
                    logger.info(f"{key}: {value}")
            else:
                logger.warning("No data retrieved from Congress.gov API")
        except Exception as e:
            logger.error(f"Error using Congress.gov API: {e}")
            logger.info("Falling back to web scraping method")
            use_congress_api = False

    # Continue with existing web scraping logic if not using API or API failed
    if not use_congress_api:
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
            vacancies_df = fetch_data("vacancies", current_year, years_back)
            if not vacancies_df.empty:
                dfs.append(vacancies_df)
                logger.info(f"Processed {len(vacancies_df)} vacancy records")

            # Fetch and process confirmations
            logger.info("Processing confirmation data...")
            confirmations_df = fetch_data("confirmations", current_year, years_back)
            if not confirmations_df.empty:
                dfs.append(confirmations_df)
                logger.info(f"Processed {len(confirmations_df)} confirmation records")

            # Fetch and process emergencies
            logger.info("Processing judicial emergency data...")
            emergencies_df = fetch_data("emergencies", current_year, years_back)
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
