"""
Helper functions for the feature engineering notebook
"""

from pathlib import Path
import re
from typing import Dict

from loguru import logger
import pandas as pd

from nomination_predictor.congress_api_utils import (
    clean_name,
    create_full_name_from_parts,
    enrich_congress_nominees_dataframe,
)


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
        fjc_service = pd.read_csv(raw_data_dir / "federal_judicial_service.csv")
        cong_nominees = pd.read_csv(raw_data_dir / "congress_nominees_cache.csv")
        cong_nominations = pd.read_csv(raw_data_dir / "congress_nominations_cache.csv")
        
        logger.info(f"Loaded {len(fjc_judges)} judges, {len(fjc_service)} service records, "
                    f"{len(cong_nominees)} congress nominees, {len(cong_nominations)} nominations")
        
        # Enrich the nominees dataframe with name fields and court information from nominations
        enriched_nominees = enrich_congress_nominees_dataframe(cong_nominees, cong_nominations)
        
        # Enrich the FJC judges dataframe with full name fields
        enriched_judges = enrich_fjc_judges(fjc_judges)
        
        # Return all dataframes
        return {
            "fjc_judges": enriched_judges,
            "fjc_service": fjc_service,
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
    
    position_pattern = r"to be (?:a |an )?(.*?)(?: for| of| to)(?: the)? (.*?)(?:,|\.| vice)"
    state_pattern = r"of (.*?)(?:,|\.| to be)"
    
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


def analyze_match_failures(nominees_df: pd.DataFrame, threshold: int = 80) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Analyze records that didn't meet the match score threshold and provide
    explanations for why they might have failed to match.
    
    Args:
        nominees_df: The nominees dataframe with match_score column
        threshold: The match score threshold used for determining matches
        
    Returns:
        DataFrame with failure analysis information
    """
    # Identify unmatched records
    unmatched = nominees_df[nominees_df["match_score"] < threshold].copy() if "match_score" in nominees_df.columns else nominees_df.copy()
    
    if len(unmatched) == 0:
        logger.info("No unmatched records to analyze")
        return pd.DataFrame()
    
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
