"""This file contains code to create features for modeling, i.e. data which wasn't in the raw data but which is being made available for the model.
It shall never output to "raw" data.  It is meant to take raw or interim data as input and deliver interim data as output.


Examples of data feature-generation include utility functions for feature-engineering U.S. federal political timelines, or capabilities such as:

Given a vacancy is on record as having existed in a specified court, with a named incumbent judge, a known reason the vacancy occurred, and a known date the vacancy began, when a vacancy with equivalent data is found on other months, then it shall be treated as the same vacancy incident.
Given earlier records for a vacancy incident lack a nominee and nomination date, when a later record for the same vacancy incident has a nominee and nomination date, then the nominee and nomination date shall be merged onto that vacancy incident's records in interim data.
Given a vacancy is on record as having existed in a specified court, with a named incumbent judge, when a vacancy with equivalent court location and incumbent is found to have occurred with a different reason and/or vacancy date (e.g. a nomination is withdrawn and reopened), then the two vacancy incidents shall be treated as each deserving of their own unique record in interim data.
Given a nomination is on record as having occurred on a specified date, when filling in date-dependent feature data (e.g. which President performed the nomination), then the interim data shall be updated to include date-inferred data (e.g. a numeric ordinal indicator identifying that President who was in office on the date of nomination.
"""

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, Optional, Tuple

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from nomination_predictor.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from nomination_predictor.congress_api_utils import (
    clean_name,
    create_full_name_from_parts,
    enrich_congress_nominees_dataframe,
)

app = typer.Typer()

# ---------------------------------------------------------------------------
# 1.  PRESIDENTIAL TERMS  (simple flat list; None = still in office)
#    tuple = (president_number, first_inaug_date, last_day_in_office_or_None)
# ---------------------------------------------------------------------------

PRESIDENT_TERMS = [
    # Modern era (you can extend backward easily)
    (46, date(2021, 1, 20), None),            # Joseph R. Biden
    (45, date(2017, 1, 20), date(2021, 1, 20)),
    (44, date(2009, 1, 20), date(2017, 1, 20)),
    (43, date(2001, 1, 20), date(2009, 1, 20)),
    (42, date(1993, 1, 20), date(2001, 1, 20)),
    (41, date(1989, 1, 20), date(1993, 1, 20)),
    (40, date(1981, 1, 20), date(1989, 1, 20)),
    (39, date(1977, 1, 20), date(1981, 1, 20)),
    (38, date(1974, 8,  9), date(1977, 1, 20)),
    (37, date(1969, 1, 20), date(1974, 8,  9)),
    (36, date(1963, 11,22), date(1969, 1, 20)),
    (35, date(1961, 1, 20), date(1963, 11,22)),
    (34, date(1953, 1, 20), date(1961, 1, 20)),
    (33, date(1945, 4, 12), date(1953, 1, 20)),
    (32, date(1933, 3,  4), date(1945, 4, 12)),
    (31, date(1929, 3,  4), date(1933, 3,  4)),
    (30, date(1923, 8,  2), date(1929, 3,  4)),
    (29, date(1921, 3,  4), date(1923, 8,  2)),
    (28, date(1913, 3,  4), date(1921, 3,  4)),
    (27, date(1909, 3,  4), date(1913, 3,  4)),
    (26, date(1901, 9, 14), date(1909, 3,  4)),
    #  … extend farther back as needed …
]

# Pre-sort newest-first so we can break quickly in lookups
PRESIDENT_TERMS.sort(key=lambda t: t[1], reverse=True)


# ---------------------------------------------------------------------------
# 2.  BASIC HELPERS
# ---------------------------------------------------------------------------

def _election_day(year: int) -> date:
    """
    Return the U.S. general Election Day for a given year:
    'the Tuesday following the first Monday in November'.
    """
    # First Monday in November
    first_monday = date(year, 11, 1)
    while first_monday.weekday() != 0:   # 0 = Monday
        first_monday += timedelta(days=1)
    # Tuesday following that
    return first_monday + timedelta(days=1)


@lru_cache(None)
def _president_record(d: date) -> tuple[int, date]:
    """
    Return (president_number, first_inaug_date_of_that_presidency)
    for the date supplied.  Raises ValueError if out of range.
    """
    for number, start, end in PRESIDENT_TERMS:
        if d >= start and (end is None or d < end):
            return number, start
    raise ValueError(f"Date {d} out of president table range.")


# ---------------------------------------------------------------------------
# 3.  PUBLIC API FUNCTIONS
# ---------------------------------------------------------------------------

def president_number(d: date) -> int:
    """38 for Gerald Ford, 46 for Joe Biden, …"""
    return _president_record(d)[0]


def presidential_term_index(d: date) -> int:
    """
    1 for the first four-year term of that president,
    2 for the second, 3/4 for FDR, etc.
    """
    number, first_inaug = _president_record(d)
    # Years (minus a day fudge) elapsed since first inauguration
    years_elapsed = (d - first_inaug).days // 365.2425
    return int(years_elapsed // 4) + 1


def days_into_current_term(d: date) -> int:
    """1-based: inauguration day itself returns 1."""
    number, first_inaug = _president_record(d)
    term_idx = presidential_term_index(d)
    term_start = first_inaug.replace(year=first_inaug.year + 4*(term_idx-1))
    return (d - term_start).days + 1


def days_until_next_presidential_election(d: date) -> int:
    """
    Count of days from `d` (exclusive) up to the **next** presidential-year
    Election Day (years divisible by 4). Returns 0 if `d` *is* Election Day.
    """
    year = d.year
    while year % 4 != 0 or d > _election_day(year):
        year += 1
    return (_election_day(year) - d).days


def days_until_next_midterm_election(d: date) -> int:
    """
    Days until the next even-year Election Day **that is not a presidential year**.
    """
    year = d.year
    while year % 2 != 0 or year % 4 == 0 or d > _election_day(year):
        year += 1
    return (_election_day(year) - d).days


# ----------------  CONGRESS & SESSION  -------------------------------------

CONGRESS_FIRST_YEAR = 1789   # First Congress began Mar-4-1789
CONGRESS_START_MONTH_PRE20TH = 3   # March 4 before 1935
CONGRESS_START_MONTH_20TH    = 1   # Jan   3 starting 1935

def congress_number(d: date) -> int:
    """
    118  → Jan-3-2023 to Jan-3-2025, etc.
    Formula: each Congress spans two calendar years, starting in odd years.
    """
    if d < date(1789, 3, 4):
        raise ValueError("Congress did not yet exist.")
    start_year = d.year if d.year % 2 else d.year - 1
    return (start_year - CONGRESS_FIRST_YEAR) // 2 + 1


def congress_session(d: date) -> int:
    """
    1  → first (odd-year) session
    2  → second (even-year) session
    3  → anything else (special/emergency)
    """
    cong_start_year = d.year if d.year % 2 else d.year - 1
    if d.year == cong_start_year:
        return 1
    elif d.year == cong_start_year + 1:
        return 2
    else:
        return 3
    
    
def self_test():
    """Output logging statements describing today's date as a sort of minimal self-test"""
    today = date.today()
    logger.info("Today:", today)
    logger.info("President #: ", president_number(today))
    logger.info("Term index : ", presidential_term_index(today))
    logger.info("Days into term:", days_into_current_term(today))
    logger.info("Days until next midterm :", days_until_next_midterm_election(today))
    logger.info("Days until next pres.  :", days_until_next_presidential_election(today))
    logger.info("Congress # :", congress_number(today))
    logger.info("Session    :", congress_session(today))


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = INTERIM_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names in a DataFrame: convert to lowercase and replace spaces with underscores.
    
    Args:
        df: DataFrame to normalize
        
    Returns:
        DataFrame with normalized column names
    """
    if df is None or df.empty:
        return df
        
    # Create a mapping of old names to new names
    column_mapping = {col: col.casefold().replace(' ', '_') for col in df.columns}
    
    # Rename columns using the mapping
    return df.rename(columns=column_mapping)


def normalize_all_dataframes(dataframes_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Normalize column names for all DataFrames in a dictionary.
    
    Args:
        dataframes_dict: Dictionary of DataFrames with string keys
        
    Returns:
        Dictionary with same keys but DataFrames having normalized column names
    """
    normalized_dfs: Dict[str, pd.DataFrame] = {}
    
    for name, df in dataframes_dict.items():
        if df is not None and not df.empty:
            normalized_dfs[name] = normalize_dataframe_columns(df)
        else:
            normalized_dfs[name] = df
    
    return normalized_dfs


def enrich_fjc_judges(judges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived name fields to the FJC judges DataFrame.
    
    Args:
        judges_df: FJC judges DataFrame with first_name, middle_name, last_name columns
        
    Returns:
        DataFrame with additional name fields
    """
    # Create a copy to avoid modifying the original
    enriched_df = judges_df.copy()
    
    # Check for column names - they might be differently named
    first_col = next((col for col in enriched_df.columns if 'first' in col.lower()), None)
    middle_col = next((col for col in enriched_df.columns if 'middle' in col.lower()), None)
    last_col = next((col for col in enriched_df.columns if 'last' in col.lower()), None)
    suffix_col = next((col for col in enriched_df.columns if 'suffix' in col.lower()), None)
    
    if first_col and last_col:  # Minimum required fields
        # Create full name from components
        enriched_df["name_full"] = enriched_df.apply(
            lambda row: create_full_name_from_parts(
                row.get(first_col), 
                row.get(middle_col) if middle_col else None, 
                row.get(last_col),
                row.get(suffix_col) if suffix_col else None
            ), 
            axis=1
        )
        
        # Add cleaned full name
        enriched_df["full_name_clean"] = enriched_df["name_full"].apply(clean_name)
    
    return enriched_df


def load_and_prepare_dataframes(raw_data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load and prepare dataframes needed for feature engineering.
    
    Args:
        raw_data_dir: Path to raw data directory
        
    Returns:
        Dictionary of prepared dataframes
    """
    try:
        # Load raw data
        fjc_judges = pd.read_csv(raw_data_dir / "judges.csv")
        fjc_federal_judicial_service = pd.read_csv(raw_data_dir / "federal_judicial_service.csv")
        fjc_demographics = pd.read_csv(raw_data_dir / "demographics.csv")
        fjc_education = pd.read_csv(raw_data_dir / "education.csv")
        fjc_other_federal_judicial_service = pd.read_csv(raw_data_dir / "other_federal_judicial_service.csv")
        fjc_other_nominations_recess = pd.read_csv(raw_data_dir / "other_nominations_recess.csv")
        seat_timeline = pd.read_csv(raw_data_dir / "seat_timeline.csv")
        cong_nominees = pd.read_csv(raw_data_dir / "nominees.csv")
        cong_nominations = pd.read_csv(raw_data_dir / "nominations.csv")
        
        logger.info(f"Loaded {len(fjc_judges)} judges, {len(fjc_federal_judicial_service)} service records, "
                    f"{len(cong_nominees)} congress nominees, {len(cong_nominations)} nominations")
        

        # Collect all dataframes into a dictionary
        all_dfs = {
            "fjc_judges": fjc_judges,
            "fjc_federal_judicial_service": fjc_federal_judicial_service,
            "fjc_demographics": fjc_demographics,
            "fjc_education": fjc_education,
            "fjc_other_federal_judicial_service": fjc_other_federal_judicial_service,
            "fjc_other_nominations_recess": fjc_other_nominations_recess,
            "seat_timeline": seat_timeline,
            "cong_nominees": cong_nominees,
            "cong_nominations": cong_nominations
        }
        
        # Normalize column names for all dataframes
        normalized_dfs = normalize_all_dataframes(all_dfs)
        
        # Extract the normalized dataframes
        fjc_judges = normalized_dfs["fjc_judges"]
        fjc_federal_judicial_service = normalized_dfs["fjc_federal_judicial_service"]
        fjc_demographics = normalized_dfs["fjc_demographics"]
        fjc_education = normalized_dfs["fjc_education"]
        fjc_other_federal_judicial_service = normalized_dfs["fjc_other_federal_judicial_service"]
        fjc_other_nominations_recess = normalized_dfs["fjc_other_nominations_recess"]
        seat_timeline = normalized_dfs["seat_timeline"]
        cong_nominees = normalized_dfs["cong_nominees"]
        cong_nominations = normalized_dfs["cong_nominations"]
        
        # Enrich the nominees dataframe with name fields and court information from nominations
        enriched_nominees = enrich_congress_nominees_dataframe(cong_nominees, cong_nominations)
        
        # Enrich the FJC judges dataframe with full name fields
        enriched_judges = enrich_fjc_judges(fjc_judges)
        
        # Return all dataframes
        return {
            "fjc_judges": enriched_judges,
            "fjc_federal_judicial_service": fjc_federal_judicial_service,
            "fjc_demographics": fjc_demographics,
            "fjc_education": fjc_education,
            "fjc_other_federal_judicial_service": fjc_other_federal_judicial_service,
            "fjc_other_nominations_recess": fjc_other_nominations_recess,
            "seat_timeline": seat_timeline,
            "cong_nominees": enriched_nominees,
            "cong_nominations": cong_nominations
        }
    except Exception as e:
        logger.error(f"Error loading dataframes: {e}")
        raise


def extract_court_and_position(nominations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract court and position information from nomination descriptions.
    
    Args:
        nominations_df: Dataframe containing nomination data with 'description' field
        
    Returns:
        DataFrame with additional extracted fields
    """
    result_df = nominations_df.copy()
    
    # Extract court and position information from descriptions
    courts = []
    positions = []
    states = []
    
    position_pattern = r"to be (?:a |an )?(.*?)(?: for| of| to)(?: the)? (.*?)(?:,|\.|\s+vice)"
    state_pattern = r"of (.*?)(?:,|\.|\s+to be)"
    
    for desc in result_df["description"]:
        if pd.isna(desc):
            courts.append("")
            positions.append("")
            states.append("")
            continue
            
        # Try to extract position and court
        pos_match = re.search(position_pattern, desc)
        if pos_match:
            position = pos_match.group(1).strip()
            court = pos_match.group(2).strip()
            positions.append(position)
            courts.append(court)
        else:
            positions.append("")
            courts.append("")
            
        # Try to extract state
        state_match = re.search(state_pattern, desc)
        if state_match:
            state = state_match.group(1).strip()
            states.append(state)
        else:
            states.append("")
    
    # Add extracted fields to dataframe
    result_df["extracted_position"] = positions
    result_df["extracted_court"] = courts
    result_df["extracted_state"] = states
    
    return result_df


def analyze_match_failures(nominees_df: pd.DataFrame, threshold: int = 80) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Analyze records that didn't meet the match score threshold and provide
    explanations for why they might have failed to match.
    
    Args:
        nominees_df: The nominees dataframe with match_score column
        threshold: The match score threshold used for determining matches
        
    Returns:
        Tuple of (unmatched_df, reason_counts, examples)
    """
    # Identify unmatched records
    unmatched = nominees_df[nominees_df["match_score"] < threshold].copy() if "match_score" in nominees_df.columns else nominees_df.copy()
    
    if len(unmatched) == 0:
        logger.info("No unmatched records to analyze")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Add failure reason column
    reasons = []
    for _, row in unmatched.iterrows():
        if "match_score" not in nominees_df.columns:
            reasons.append("No match_score column present in dataframe")
            continue
            
        if pd.isna(row.get("match_score")):
            reasons.append("No match attempt was made (match_score is NaN)")
        elif row["match_score"] == 0:
            reasons.append("No potential match candidates found")
        elif row["match_score"] < 50:
            reasons.append("Very low similarity - likely different person")
        else:
            # Score between 50 and threshold
            reasons.append(f"Marginal match (score {row['match_score']:.1f}) - check name and court")
    
    unmatched["failure_reason"] = reasons
    
    # Count occurrences of each failure reason
    reason_counts = unmatched["failure_reason"].value_counts().reset_index()
    reason_counts.columns = ["Failure Reason", "Count"]
    
    # Get examples for each failure reason
    examples = {}
    for reason in reason_counts["Failure Reason"].unique():
        examples[reason] = unmatched[unmatched["failure_reason"] == reason].head(3)[["full_name", "court_clean", "match_score", "failure_reason"]]
    
    return unmatched, reason_counts, examples


def is_nominee_confirmed(nominee_row: pd.Series, nominations_df: pd.DataFrame, citation_key: str = "citation") -> bool:
    """
    Determine if a nominee was confirmed based on their latestaction in the nominations table.
    
    Args:
        nominee_row: Row from nominees DataFrame containing citation
        nominations_df: DataFrame with nomination information including latestaction field
        citation_key: Column name for the citation in both DataFrames
        
    Returns:
        Boolean indicating whether the nominee was confirmed
    """
    if citation_key not in nominee_row or pd.isna(nominee_row[citation_key]):
        return False
        
    citation = nominee_row[citation_key]
    nomination_row = nominations_df[nominations_df[citation_key] == citation]
    
    if nomination_row.empty:
        return False
        
    # Get the latestaction field which contains a dict with 'actionDate' and 'text'
    latest_action = nomination_row.iloc[0].get("latestaction")
    
    if not latest_action or not isinstance(latest_action, dict):
        return False
        
    # Check if the text contains 'Confirmed'
    action_text = latest_action.get("text", "")
    return "Confirmed" in action_text


def filter_confirmed_nominees(nominees_df: pd.DataFrame, nominations_df: pd.DataFrame, citation_key: str = "citation") -> pd.DataFrame:
    """
    Filter the nominees DataFrame to only include confirmed nominees.
    
    Args:
        nominees_df: DataFrame with nominee information
        nominations_df: DataFrame with nomination information including latestaction field
        citation_key: Column name for the citation in both DataFrames
        
    Returns:
        DataFrame containing only confirmed nominees
    """
    # Apply the confirmation check to each row
    is_confirmed = nominees_df.apply(
        lambda row: is_nominee_confirmed(row, nominations_df, citation_key),
        axis=1
    )
    
    # Filter the DataFrame
    confirmed_nominees = nominees_df[is_confirmed].copy()
    logger.info(f"Filtered {len(nominees_df)} nominees to {len(confirmed_nominees)} confirmed nominees")
    
    return confirmed_nominees


def merge_nominees_with_nominations(
    nominees_df: pd.DataFrame, 
    nominations_df: pd.DataFrame,
    join_key: str = "citation"
) -> pd.DataFrame:
    """
    Merge nominees with their corresponding nominations.
    
    Args:
        nominees_df: Dataframe with nominee information
        nominations_df: Dataframe with nomination information
        join_key: Key to use for joining the dataframes
        
    Returns:
        Merged dataframe
    """
    if join_key not in nominees_df.columns or join_key not in nominations_df.columns:
        raise ValueError(f"Join key '{join_key}' not found in both dataframes")
    
    # Extract court and position information
    enhanced_nominations = extract_court_and_position(nominations_df)
    
    # Merge dataframes
    merged = nominees_df.merge(
        enhanced_nominations,
        on=join_key,
        how="left",
        suffixes=("_nominee", "_nomination")
    )
    
    return merged


def filter_non_judicial_nominations(
    nominations_df: pd.DataFrame, 
    nominees_df: pd.DataFrame,
    non_judicial_titles: list[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out non-judicial nominations based on position titles.
    
    Args:
        nominations_df: DataFrame containing nomination records with 'nominee_positiontitle' and 'citation' columns
        nominees_df: DataFrame containing nominee records with 'citation' column
        non_judicial_titles: List of strings indicating non-judicial position titles
        
    Returns:
        Tuple of (filtered_nominations_df, filtered_nominees_df)
    """
    if non_judicial_titles is None:
        non_judicial_titles = [
            "Attorney", "Board", "Commission", "Director", "Marshal",
            "Assistant", "Representative", "Secretary of", "Member of"
        ]
    
    # Make copies to avoid SettingWithCopyWarning
    nominations = nominations_df.copy()
    nominees = nominees_df.copy()
    
    # Find citations of rows with non-judicial titles
    non_judicial_mask = nominations["nominee_positiontitle"].str.contains(
        '|'.join(non_judicial_titles), 
        na=False
    )
    citations_to_drop = nominations.loc[non_judicial_mask, "citation"].unique()
    
    # Log the number of non-judicial nominations being removed
    logger.info(f"Found {len(citations_to_drop)} unique citations with non-judicial titles")
    
    # Filter out the non-judicial nominations
    filtered_nominations = nominations[~nominations["citation"].isin(citations_to_drop)]
    filtered_nominees = nominees[~nominees["citation"].isin(citations_to_drop)]
    
    # Log the results
    logger.info(
        f"Removed {len(nominations) - len(filtered_nominations)}/{len(nominations)} "
        f"non-judicial nominations and {len(nominees) - len(filtered_nominees)}/{len(nominees)} "
        "corresponding nominee records"
    )
    
    return filtered_nominations, filtered_nominees


if __name__ == "__main__":
    app()
