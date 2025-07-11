"""
Functions for parsing and processing Federal Judicial Center (FJC) data.

This module provides utilities for loading, parsing, and transforming FJC data files,
building the seat timeline table, and crosswalking with other data sources.
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger
import pandas as pd

from nomination_predictor.config import EXTERNAL_DATA_DIR

# Path constants
FJC_DATA_DIR = EXTERNAL_DATA_DIR / "FederalJudicialCenter"  # Source directory for FJC data


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


def load_fjc_csv(file_name: str, date_columns: List[str] = None) -> pd.DataFrame:
    """
    Load a FJC CSV file with proper date parsing and data types.
    
    Args:
        file_name: Name of file in the FJC data directory
        date_columns: List of column names containing dates
        
    Returns:
        DataFrame with properly parsed data
    """
    full_path = FJC_DATA_DIR / file_name
    
    if not full_path.exists():
        logger.error(f"FJC data file not found: {full_path}")
        raise FileNotFoundError(f"File not found: {full_path}")
    
    logger.info(f"Loading FJC data file: {file_name}")
    
    # First read without date parsing to get column names
    df = pd.read_csv(full_path)
    
    # Identify date columns if not specified
    if date_columns is None:
        date_columns = [col for col in df.columns if 'date' in col.lower()]
    
    logger.debug(f"Parsing date columns: {date_columns}")
    
    # Apply date parsing to identified columns
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_fjc_date)
    
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
    
    # Select and rename relevant columns
    seat_timeline = service_df[['nid', 'seat_id', 'court', 
                               'commission_date', 'termination_date',
                               'termination_reason']].copy()
    
    # Sort by seat_id and commission_date
    seat_timeline = seat_timeline.sort_values(['seat_id', 'commission_date'])
    
    # Create vacancy date (same as termination_date for most cases)
    seat_timeline['vacancy_date'] = seat_timeline['termination_date']
    
    # Handle special case: "appointment to same court" shouldn't create vacancy
    mask = seat_timeline['termination_reason'].str.contains(
        'appointment to same court', case=False, na=False)
    seat_timeline.loc[mask, 'vacancy_date'] = pd.NaT
    
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
                        f"Fixed termination date > successor commission for seat {seat_id}: "
                        f"{termination_dates[i]} > {commission_dates[i+1]}"
                    )
                    seat_timeline.loc[idx, 'termination_date'] = commission_dates[i+1]
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
    lookup = seat_timeline[['seat_id', 'court', 'nid', 'vacancy_date']].copy()
    
    # Filter out rows without vacancy dates
    lookup = lookup[pd.notna(lookup['vacancy_date'])]
    
    # Rename for clarity
    lookup = lookup.rename(columns={'nid': 'predecessor_nid'})
    
    return lookup


def load_judges_data(include_demographics: bool = True) -> pd.DataFrame:
    """
    Load and process the judges.csv file with optional demographics data.
    
    Args:
        include_demographics: Whether to join with demographics.csv
        
    Returns:
        DataFrame with judge information
    """
    logger.info("Loading judges data")
    
    # Load judges.csv
    judges_df = load_fjc_csv('judges.csv')
    
    if include_demographics:
        # Load demographics.csv
        try:
            demo_df = load_fjc_csv('demographics.csv')
            # Join on nid
            judges_df = judges_df.merge(demo_df, on='nid', how='left')
            logger.info("Added demographic information to judges data")
        except FileNotFoundError:
            logger.warning("Demographics file not found, proceeding without demographics")
    
    return judges_df


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
    
    # Create a mapping of predecessor name to nid using judges_df
    name_to_nid = {}
    for _, row in judges_df.iterrows():
        full_name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
        if full_name:
            name_to_nid[full_name] = row['nid']
    
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
    
    # Map predecessor names to nids
    nomination_df['predecessor_nid'] = nomination_df['predecessor_name'].apply(find_nid_by_name)
    
    # Join with seat_timeline based on predecessor_nid
    result_df = pd.merge(
        nomination_df,
        seat_timeline[['nid', 'seat_id', 'court']],
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
