"""
Functions for parsing and processing Federal Judicial Center (FJC) data.

This module provides utilities for loading, parsing, and transforming FJC data files,
building the seat timeline table, and crosswalking with other data sources.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm

from nomination_predictor.config import EXTERNAL_DATA_DIR

# Path constants
FJC_DATA_DIR = EXTERNAL_DATA_DIR / "FederalJudicialCenter"  # Source directory for FJC data

# FJC data file URLs from https://www.fjc.gov/history/judges/biographical-directory-article-iii-federal-judges-export
FJC_DATA_URLS = {
    'judges.csv': 'https://www.fjc.gov/sites/default/files/history/judges.csv',
    'demographics.csv': 'https://www.fjc.gov/sites/default/files/history/demographics.csv',
    'federal-judicial-service.csv': 'https://www.fjc.gov/sites/default/files/history/federal-judicial-service.csv',
    'education.csv': 'https://www.fjc.gov/sites/default/files/history/education.csv',
    'professional-career.csv': 'https://www.fjc.gov/sites/default/files/history/professional-career.csv',
    'other-nominations-recess.csv': 'https://www.fjc.gov/sites/default/files/history/other-nominations-recess.csv',
    'other-federal-judicial-service.csv':' https://www.fjc.gov/sites/default/files/history/other-federal-judicial-service.csv',
    'judges.xlsx': 'https://www.fjc.gov/sites/default/files/history/judges.xlsx',
    'categories.xlsx': 'https://www.fjc.gov/sites/default/files/history/categories.xlsx',
}

# Required FJC data files (subset of FJC_DATA_URLS)
REQUIRED_FJC_FILES = [
    'demographics.csv',
    'education.csv',
    'federal-judicial-service.csv',
    'judges.csv',
    'other-nominations-recess.csv',
    'other-federal-judicial-service.csv',
    'professional-career.csv',
]

# Optional FJC data files (subset of FJC_DATA_URLS)
OPTIONAL_FJC_FILES = [
    'judges.xlsx',
    'categories.xlsx',
]


def parse_fjc_date(date_str: str) -> Optional[pd.Timestamp]:
    """
    Parse dates from FJC data files, handling various formats and pre-1900 dates.
    
    FJC dates can be:
    - Excel: 'yyyy-mm-dd' string format (especially for pre-1900 dates)
    - CSV: 'mm/dd/yyyy' format
    - Empty strings or NaN values
    
    Args:
        date_str: Date string in 'yyyy-mm-dd' or 'mm/dd/yyyy' format, or empty
        
    Returns:
        pd.Timestamp or pd.NaT if parsing fails
    """
    if not date_str or pd.isna(date_str):
        return pd.NaT
    
    date_str = str(date_str).strip()
    if not date_str:
        return pd.NaT
    
    # Try standard pandas date parsing first
    return_val = pd.to_datetime(date_str, errors='coerce')
    if pd.notna(return_val):
        return return_val
    
    # If standard parsing fails, try manual parsing
    
    # Handle yyyy-mm-dd format (Excel format for pre-1900 dates)
    if re.match(r'^\d{4}-\d{1,2}-\d{1,2}$', date_str):
        try:
            year, month, day = map(int, date_str.split('-'))
            return pd.Timestamp(year=year, month=month, day=day)
        except (ValueError, IndexError):
            return pd.NaT
    
    # Handle mm/dd/yyyy format (CSV format)
    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_str):
        try:
            month, day, year = map(int, date_str.split('/'))
            return pd.Timestamp(year=year, month=month, day=day)
        except (ValueError, IndexError):
            return pd.NaT
    
    return pd.NaT


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names by casefolding and replacing spaces with underscores.
    
    Args:
        df: DataFrame with columns to normalize
        
    Returns:
        DataFrame with normalized column names
    """
    # Create a column mapper: old_name -> new_name
    column_map = {
        col: col.lower().replace(' ', '_') for col in df.columns
    }
    
    # Rename columns
    return df.rename(columns=column_map)


def load_fjc_csv(file_name: str, normalize: bool = True, date_columns: List[str] = None) -> pd.DataFrame:
    """
    Load a FJC CSV file with optional column normalization and proper date parsing.
    
    Args:
        file_name: Name of file in the FJC data directory
        normalize: Whether to normalize column names (casefold, replace spaces with underscores)
        date_columns: List of column names containing dates (or None to auto-detect)
        
    Returns:
        DataFrame with properly parsed data
    """
    # Ensure we're dealing with string paths for consistency
    if isinstance(file_name, str):
        full_path = os.path.join(FJC_DATA_DIR, file_name)
    else:  # Assume it's a Path object
        full_path = file_name
    
    logger.info(f"Loading FJC data file: {file_name}")
    
    # First read to get structure
    df = pd.read_csv(full_path)
    
    # Apply column normalization if requested
    if normalize:
        df = normalize_columns(df)
        
    # Auto-detect date columns if not specified
    if date_columns is None:
        # Heuristic: Look for column names containing 'date'
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        logger.debug(f"Auto-detected date columns: {date_columns}")
    
    # If we have date columns to parse
    if date_columns and len(date_columns) > 0:
        # Use converters for date parsing
        converters = {col: parse_fjc_date for col in date_columns if col in df.columns}
        
        if converters:  # Only re-read if we have valid converters
            # Re-read with date parsing
            df = pd.read_csv(full_path, converters=converters)
            
            # Apply normalization again if needed
            if normalize:
                df = normalize_columns(df)
    
    return df


def build_seat_timeline(service_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the seat timeline table from the federal-judicial-service data.
    
    The seat timeline is one row per incumbent-seat tenure, forming the
    spine of our dataset.
    
    Args:
        service_df: DataFrame from federal-judicial-service.csv
        
    Returns:
        DataFrame with seat timeline
    """
    logger.info("Building seat timeline table")
    
    # Keep all columns from the original DataFrame - defer cleaning to later notebooks
    seat_timeline = service_df.copy()
    
    # Check if the expected columns exist in the DataFrame
    required_columns = [
        'nid', 'seat_id', 'commission_date', 'termination_date', 'termination'
    ]
    missing_columns = [col for col in required_columns if col not in seat_timeline.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in service_df: {missing_columns}")
    
    # Sort by seat ID and commission date
    seat_timeline = seat_timeline.sort_values(['seat_id', 'commission_date'])
    
    # Create vacancy date (same as termination_date for most cases)
    # Add as a new column without modifying original data
    seat_timeline['vacancy_date'] = seat_timeline['termination_date']
    
    # Note: We're deferring special-case handling (such as analyzing different termination reasons)
    # to later data cleaning and analysis stages, not in this basic ingestion function
    
    # Check for termination date > next commission date (the Dec 2023 fix mentioned)
    # This requires looking at next row for each seat
    seat_ids = seat_timeline['seat_id'].unique()
    
    for seat_id in seat_ids:
        seat_rows = seat_timeline[seat_timeline['seat_id'] == seat_id]
        
        if len(seat_rows) <= 1:
            continue
        
        # Get commission dates and termination dates
        commission_dates = seat_rows['commission_date'].tolist()
        termination_dates = seat_rows['termination_date'].tolist()
        
        # Check and fix each termination date
        for i in range(len(seat_rows) - 1):
            if pd.notna(termination_dates[i]) and pd.notna(commission_dates[i+1]):
                if termination_dates[i] > commission_dates[i+1]:
                    # Fix: set termination date to commission date of successor
                    idx = seat_rows.iloc[i].name
                    logger.warning(
                        f"Editing a derived vacancy_date: Termination date > successor commission for seat {seat_id}: "
                        f"{termination_dates[i]} > {commission_dates[i+1]}"
                    )
                    # Update both termination date and vacancy date
                    # Note: We're not changing the original Termination Date column,
                    # just our derived vacancy_date column
                    seat_timeline.loc[idx, 'vacancy_date'] = commission_dates[i+1]
    
    return seat_timeline


def get_predecessor_info(seat_timeline: pd.DataFrame) -> pd.DataFrame:
    """
    Create a lookup table mapping vacancy dates to predecessor information.
    
    This helps with crosswalking AOUSC vacancies and Congress.gov nominations.
    
    Args:
        seat_timeline: DataFrame from build_seat_timeline()
        
    Returns:
        DataFrame with seat_id, vacancy_date, and predecessor_nid
    """
    logger.info("Building seat predecessor lookup table")
    
    # Select relevant columns
    lookup = seat_timeline[['seat_id', 'nid', 'vacancy_date']].copy()
    
    # Filter out rows without vacancy dates
    lookup = lookup[pd.notna(lookup['vacancy_date'])]
    
    # Rename for clarity
    lookup = lookup.rename(columns={'nid': 'predecessor_nid'})
    
    return lookup


def download_fjc_file(file_name: str, force: bool = False) -> bool:
    """
    Download a single FJC data file if it doesn't exist or force is True.
    
    Args:
        file_name: Name of the file to download (must be in FJC_DATA_URLS)
        force: If True, download even if the file already exists
        
    Returns:
        True if the file was downloaded or already exists, False otherwise
    """
    # Ensure we know the URL for this file
    if file_name not in FJC_DATA_URLS:
        logger.error(f"Unknown FJC data file: {file_name}")
        return False
    
    # Create the destination path
    dest_path = FJC_DATA_DIR / file_name
    
    # Check if the file already exists
    if dest_path.exists() and not force:
        logger.debug(f"FJC data file already exists: {file_name}")
        return True
    
    # Create the directory if it doesn't exist
    os.makedirs(FJC_DATA_DIR, exist_ok=True)
    
    # Get the URL for this file
    url = FJC_DATA_URLS[file_name]
    
    # Download the file
    try:
        logger.info(f"Downloading FJC data file: {file_name} from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        
        # Get the total file size for the progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        # Download with progress bar
        with open(dest_path, 'wb') as file, tqdm(
                desc=f"Downloading {file_name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
        
        logger.info(f"Downloaded FJC data file: {file_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download FJC data file {file_name}: {str(e)}")
        return False


def ensure_fjc_data_files(required_only: bool = False) -> Tuple[List[str], List[str]]:
    """
    Check for FJC data files and download any missing files.
    
    Args:
        required_only: If True, only download required files and ignore optional files
        
    Returns:
        Tuple of (downloaded_files, failed_files)
    """
    logger.info("Ensuring FJC data files are available")
    
    # Initialize results
    downloaded = []
    failed = []
    
    # Check and download required files
    for file_name in REQUIRED_FJC_FILES:
        if not (FJC_DATA_DIR / file_name).exists():
            success = download_fjc_file(file_name)
            if success:
                downloaded.append(file_name)
            else:
                failed.append(file_name)
    
    # Check and download optional files if requested
    if not required_only:
        for file_name in OPTIONAL_FJC_FILES:
            if not (FJC_DATA_DIR / file_name).exists():
                success = download_fjc_file(file_name)
                if success:
                    downloaded.append(file_name)
                else:
                    failed.append(file_name)
    
    return downloaded, failed


def load_fjc_data(auto_download: bool = True) -> dict[str, pd.DataFrame]:
    """
    Load all FJC CSV files into separate dataframes with normalized column names.
    
    Each CSV is loaded into its own DataFrame with normalized column names.
    No joining is performed at this stage; that should be done explicitly in the notebook.
    
    Args:
        auto_download: If True, automatically download any missing files
    
    Returns:
        Dict of DataFrames with keys for each CSV loaded ('judges', 'demographics', etc.)
    """
    logger.info("Loading FJC data files")
    
    # Ensure we have the required data files
    if auto_download:
        downloaded, failed = ensure_fjc_data_files(required_only=True)
        if downloaded:
            logger.info(f"Downloaded {len(downloaded)} missing FJC data files")
        if failed:
            logger.warning(f"Failed to download {len(failed)} FJC data files: {failed}")
    
    # Initialize result dictionary
    result = {}
    
    # List of core FJC files to load (prioritize required files)
    fjc_files = REQUIRED_FJC_FILES
    
    # Load each CSV file
    for file_name in fjc_files:
        try:
            # Create a key without the .csv extension
            key = file_name.replace('.csv', '').replace('-', '_')
            result[key] = load_fjc_csv(file_name)
            logger.info(f"Loaded {key} data with {len(result[key])} records")
        except FileNotFoundError:
            logger.warning(f"{file_name} not found, proceeding without it")
            continue
    
    return result





def crosswalk_congress_api(
    nomination_data: List[Dict[str, Any]], 
    seat_timeline: pd.DataFrame,
    judges_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Map Congress.gov API nomination data to FJC seat timeline.
    
    Args:
        nomination_data: List of nomination records from Congress.gov API
        seat_timeline: DataFrame from build_seat_timeline()
        judges_df: DataFrame from load_judges_data()
        
    Returns:
        DataFrame with nominations matched to seat_ids
    """
    logger.info("Crosswalking Congress.gov API data with FJC seat timeline")
    
    # Convert nomination data to DataFrame if needed
    if not isinstance(nomination_data, pd.DataFrame):
        nomination_df = pd.DataFrame(nomination_data)
    else:
        nomination_df = nomination_data.copy()
    
    # Extract predecessor name from description when available
    def extract_predecessor(desc: str) -> Optional[str]:
        if not desc or not isinstance(desc, str):
            return None
        
        # Common patterns: "vice John Doe", "succeeding Jane Smith"
        vice_match = re.search(r'vice\s+([^.,]+)', desc, re.IGNORECASE)
        if vice_match:
            return vice_match.group(1).strip()
        
        succeed_match = re.search(r'succeeding\s+([^.,]+)', desc, re.IGNORECASE)
        if succeed_match:
            return succeed_match.group(1).strip()
        
        return None
    
    nomination_df['predecessor_name'] = nomination_df.get('description', '').apply(extract_predecessor)
    
    # Create mappings from judges_df
    name_to_nid = {}
    name_to_seat_id = {}
    nid_to_seat_id = {}
    
    # Check if judges_df has 'Seat ID' column (case sensitive)
    has_seat_id = 'Seat ID' in judges_df.columns
    if has_seat_id:
        logger.info("Using 'Seat ID' from judges_df for improved crosswalking")
    
    for _, row in judges_df.iterrows():
        full_name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
        if full_name:
            # Always map name to nid
            nid = row.get('nid')
            if nid:
                name_to_nid[full_name] = nid
            
            # If Seat ID exists, create additional mappings
            if has_seat_id and not pd.isna(row.get('Seat ID')):
                seat_id = row.get('Seat ID')
                name_to_seat_id[full_name] = seat_id
                if nid:
                    nid_to_seat_id[nid] = seat_id
    
    # Function to find nid for a name with fuzzy matching if needed
    def find_nid_by_name(name: str) -> Optional[str]:
        if not name:
            return None
        
        # Direct match
        if name in name_to_nid:
            return name_to_nid[name]
        
        # Try last name only
        last_name = name.split()[-1] if name and ' ' in name else name
        for full_name, nid in name_to_nid.items():
            if full_name.endswith(last_name):
                return nid
                
        return None
    
    # Function to find seat_id for a name with fuzzy matching if needed
    def find_seat_id_by_name(name: str) -> Optional[str]:
        if not name or not has_seat_id:
            return None
        
        # Direct match
        if name in name_to_seat_id:
            return name_to_seat_id[name]
        
        # Try last name only
        last_name = name.split()[-1] if name and ' ' in name else name
        for full_name, seat_id in name_to_seat_id.items():
            if full_name.endswith(last_name):
                return seat_id
                
        return None
    
    # Map predecessor names to nids and seat_ids
    nomination_df['predecessor_nid'] = nomination_df['predecessor_name'].apply(find_nid_by_name)
    
    # If we have Seat ID in judges_df, add it to nomination_df
    if has_seat_id:
        nomination_df['seat_id'] = nomination_df['predecessor_name'].apply(find_seat_id_by_name)
        
        # For nominations with nid but no seat_id, try to get seat_id from nid
        for idx, row in nomination_df.iterrows():
            if pd.isna(row.get('seat_id')) and not pd.isna(row.get('predecessor_nid')):
                if row['predecessor_nid'] in nid_to_seat_id:
                    nomination_df.at[idx, 'seat_id'] = nid_to_seat_id[row['predecessor_nid']]
    
    # First identify what columns we have available in seat_timeline
    logger.info(f"Available columns in seat_timeline: {seat_timeline.columns.tolist()}")
    
    # Check if we have the expected columns or alternatives
    join_cols = ['nid']
    result_cols = ['nid']
    
    # Use 'Seat ID' if available, otherwise look for alternatives
    seat_id_col = None
    for col_candidate in ['seat_id', 'Seat ID', 'SEAT_ID']:
        if col_candidate in seat_timeline.columns:
            seat_id_col = col_candidate
            join_cols.append(seat_id_col)
            result_cols.append(seat_id_col)
            break
    
    # Use 'court' if available, otherwise look for alternatives
    court_col = None
    for col_candidate in ['court', 'Court', 'COURT']:
        if col_candidate in seat_timeline.columns:
            court_col = col_candidate
            join_cols.append(court_col)
            result_cols.append(court_col)
            break
    
    logger.info(f"Joining on columns: {join_cols}")
    
    # Determine best approach for joining based on available columns
    if 'seat_id' in nomination_df.columns and not nomination_df['seat_id'].isna().all():
        # If we successfully mapped seat_ids from judges_df to nominations, use that first
        logger.info("Using seat_id from judges_df for primary joining")
        
        # Find the appropriate seat_id column in seat_timeline
        seat_timeline_id_col = None
        for col in seat_timeline.columns:
            if col.lower() in ['seat id', 'seat_id', 'seatid']:
                seat_timeline_id_col = col
                break
        
        if seat_timeline_id_col:
            # Primary join on seat_id
            result_df = pd.merge(
                nomination_df,
                seat_timeline,
                left_on='seat_id',
                right_on=seat_timeline_id_col,
                how='left'
            )
            
            # For unmatched rows, try nid
            unmatched = result_df[pd.isna(result_df[seat_timeline_id_col])]
            if not unmatched.empty and 'predecessor_nid' in unmatched.columns:
                logger.info(f"Trying secondary join on nid for {len(unmatched)} unmatched records")
                secondary_matches = pd.merge(
                    unmatched,
                    seat_timeline,
                    left_on='predecessor_nid',
                    right_on='nid',
                    how='inner'
                )
                
                # Remove unmatched rows from result and add secondary matches
                matched_indices = result_df[~pd.isna(result_df[seat_timeline_id_col])].index
                result_df = pd.concat([result_df.loc[matched_indices], secondary_matches])
        else:
            # Fallback to nid if seat_id column not found in seat_timeline
            logger.warning("Could not find matching seat_id column in seat_timeline. Using nid instead.")
            result_df = pd.merge(
                nomination_df,
                seat_timeline,
                left_on='predecessor_nid',
                right_on='nid',
                how='left'
            )
    else:
        # Fallback to joining on nid
        logger.info("No seat_id mapping available, joining on nid only")
        result_df = pd.merge(
            nomination_df,
            seat_timeline,
            left_on='predecessor_nid',
            right_on='nid',
            how='left'
        )
    
    # Add match quality indicator
    result_df['seat_match_method'] = 'predecessor_name'
    result_df.loc[pd.isna(result_df['seat_id']), 'seat_match_method'] = 'unmatched'
    
    return result_df


def create_master_dataset(
    seat_timeline: pd.DataFrame,
    nomination_df: Optional[pd.DataFrame] = None,
    aousc_vacancies_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create the master dataset by joining seat timeline with other sources.
    
    Args:
        seat_timeline: DataFrame from build_seat_timeline()
        nomination_df: Optional DataFrame with Congress.gov API data
        aousc_vacancies_df: Optional DataFrame with AOUSC vacancy data
        
    Returns:
        Master dataset with all available information
    """
    logger.info("Creating master dataset")
    
    # Start with the seat timeline as the spine
    master_df = seat_timeline.copy()
    
    # Add nominations data if available
    if nomination_df is not None and not nomination_df.empty:
        logger.info("Adding nomination data to master dataset")
        # Join on seat_id
        master_df = pd.merge(
            master_df,
            nomination_df.drop(columns=['court'], errors='ignore'),
            on='seat_id',
            how='left'
        )
    
    # Add AOUSC vacancies data if available
    if aousc_vacancies_df is not None and not aousc_vacancies_df.empty:
        logger.info("Adding AOUSC vacancies data to master dataset")
        # TODO: Implement joining logic with AOUSC data
        pass
    
    return master_df
