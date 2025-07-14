"""
Entity matching utilities for connecting Congress.gov nomination data with FJC judicial data.
"""

from typing import Dict, List, Optional, Tuple

from loguru import logger
import pandas as pd
from tqdm import tqdm

from nomination_predictor.config import (
    AMBIGUITY_THRESHOLD,
    COURT_WEIGHT,
    DATE_WEIGHT,
    MATCH_THRESHOLD,
    NAME_WEIGHT,
)
from nomination_predictor.fuzzy_matching import (
    analyze_match_failures,
    calculate_court_similarity,
    calculate_date_similarity,
    calculate_name_similarity,
    find_matches_with_blocking,
)


def perform_fuzzy_matching(
    cong_df: pd.DataFrame, 
    fjc_df: pd.DataFrame, 
    threshold: int = int(MATCH_THRESHOLD*100), 
    name_weight: float = NAME_WEIGHT, 
    court_weight: float = COURT_WEIGHT, 
    date_weight: float = DATE_WEIGHT
) -> pd.DataFrame:
    """
    Perform fuzzy matching between Congress.gov nominations and FJC judicial service records.
    
    Args:
        cong_df: DataFrame with Congress.gov nomination data
        fjc_df: DataFrame with FJC judge data
        threshold: Minimum score to consider a match (0-100)
        name_weight: Weight for name similarity in final score
        court_weight: Weight for court similarity in final score
        date_weight: Weight for date similarity in final score
        
    Returns:
        DataFrame with match results (original Congress data plus match scores and nid)
    """
    logger.info(f"Running fuzzy matching with threshold {threshold}")
    
    # Prepare Congress data if needed
    if "last_name" not in cong_df.columns and "full_name" in cong_df.columns:
        from nomination_predictor.features import extract_last_name
        logger.info("Extracting last names from full names for Congress data")
        cong_df = cong_df.copy()
        cong_df["last_name_from_full_name"] = cong_df["full_name"].apply(extract_last_name)
    
    # Handle column name differences between Congress and FJC data
    # This will be handled by the find_matches_with_blocking function
    
    # Perform matching using blocking for efficiency
    results = find_matches_with_blocking(
        cong_df,
        fjc_df,
        threshold=threshold,
        name_weight=name_weight,
        court_weight=court_weight,
        date_weight=date_weight
    )
    
    # Return results
    return results


def categorize_matches(match_df: pd.DataFrame, threshold: int = 80) -> Dict[str, pd.DataFrame]:
    """
    Categorize matches into high confidence, ambiguous, and unmatched.
    
    Args:
        match_df: DataFrame with match results
        threshold: Score threshold for high confidence matches
        
    Returns:
        Dictionary with categorized DataFrames
    """
    # Check if the DataFrame is empty or missing the match_score column
    if match_df.empty or "match_score" not in match_df.columns:
        logger.warning("Empty DataFrame or missing match_score column provided to categorize_matches")
        empty_df = pd.DataFrame()
        return {
            "high_confidence": empty_df,
            "medium_confidence": empty_df,
            "low_confidence": match_df.copy() if not match_df.empty else empty_df
        }
    
    # High confidence matches (meet or exceed threshold)
    high_confidence = match_df[match_df["match_score"] >= threshold].copy()
    
    # Lower confidence but potential matches (some signal but below threshold)
    medium_confidence = match_df[
        (match_df["match_score"] < threshold) & 
        (match_df["match_score"] >= threshold * MATCH_THRESHOLD)
    ].copy()
    
    # Very low confidence or no matches
    low_confidence = match_df[match_df["match_score"] < threshold * MATCH_THRESHOLD].copy()
    
    # Sort by match score descending if any results exist
    if not high_confidence.empty:
        high_confidence = high_confidence.sort_values(by="match_score", ascending=False)
    if not medium_confidence.empty:
        medium_confidence = medium_confidence.sort_values(by="match_score", ascending=False)
    if not low_confidence.empty:
        low_confidence = low_confidence.sort_values(by="match_score", ascending=False)
    
    return {
        "high_confidence": high_confidence,
        "medium_confidence": medium_confidence,
        "low_confidence": low_confidence
    }


def find_ambiguous_matches(
    cong_df: pd.DataFrame, 
    fjc_df: pd.DataFrame, 
    threshold: int = 80, 
    ambiguity_threshold: float = AMBIGUITY_THRESHOLD
) -> pd.DataFrame:
    """
    Identify ambiguous matches where multiple FJC records have similar scores.
    
    Args:
        match_df: DataFrame with match results
        fjc_df: Original FJC DataFrame
        threshold: Score threshold for matches
        ambiguity_threshold: Factor to determine ambiguity (e.g., 0.9 means within 90% of top score)
        
    Returns:
        DataFrame with ambiguous matches and their alternatives
    """
    # Check if either DataFrame is empty or missing required columns
    if cong_df.empty or fjc_df.empty:
        logger.warning("Empty Congress DataFrame provided to find_ambiguous_matches")
        return pd.DataFrame()
    
    if "match_score" not in cong_df.columns:
        logger.warning("match_score column missing from Congress DataFrame in find_ambiguous_matches")
        return pd.DataFrame()
    
    if "nid" not in fjc_df.columns:
        logger.warning("nid column missing from FJC DataFrame in find_ambiguous_matches")
        return pd.DataFrame()
        
    ambiguous_matches = []
    
    # Only consider records that meet the threshold
    potential_matches = cong_df[cong_df["match_score"] >= threshold].copy()
    
    if potential_matches.empty:
        return pd.DataFrame()
    
    # For each match, see if there are other close candidates
    for _, cong_row in potential_matches.iterrows():
        citation = cong_row.get("citation")
        top_score = cong_row.get("match_score")
        top_nid = cong_row.get("nid")
        
        # Skip if any required values are missing
        if citation is None or top_score is None or top_nid is None:
            continue
        
        # Find all FJC records that are close matches
        close_matches = []
        
        for _, fjc_row in fjc_df.iterrows():
            # Extract data safely with fallbacks to empty strings if missing
            logger.info(f"comparing {cong_row.get('full_name_from_description', '')} to {fjc_row.get('full_name_concatenated', '')}")
            name_sim = calculate_name_similarity(cong_row.get("full_name_from_description", ""), fjc_row.get("full_name_concatenated", ""))
            court_sim = calculate_court_similarity(cong_row.get("description", ""), fjc_row.get("court_name", ""))
            date_sim = calculate_date_similarity(cong_row.get("receiveddate", ""), fjc_row.get("nomination_date", ""))
            
            # Use same weights as in original matching
            score = (name_sim * NAME_WEIGHT) + (court_sim * COURT_WEIGHT) + (date_sim * DATE_WEIGHT)
            
            fjc_nid = fjc_row.get("nid")
            # Skip if nid is missing
            if fjc_nid is None:
                continue
                
            # If score is close to the top score but not the same record
            if (score >= threshold * ambiguity_threshold) and fjc_nid != top_nid:
                close_matches.append({
                    "citation": citation,
                    "alt_nid": fjc_nid,
                    "alt_judge_name": fjc_row.get("judge_name", ""),
                    "alt_court_name": fjc_row.get("court_name", ""),
                    "alt_score": score,
                    "top_score": top_score,
                    "top_nid": top_nid,
                    "score_diff": top_score - score
                })
        
        # If we found alternative matches, this is ambiguous
        if close_matches:
            ambiguous_matches.extend(close_matches)
    
    # Convert to DataFrame
    if not ambiguous_matches:
        return pd.DataFrame()
    
    ambiguous_df = pd.DataFrame(ambiguous_matches)
    
    # Sort only if we have data and the required columns
    if not ambiguous_df.empty and "citation" in ambiguous_df.columns and "alt_score" in ambiguous_df.columns:
        return ambiguous_df.sort_values(by=["citation", "alt_score"], ascending=[True, False])
    
    return ambiguous_df


def find_unmatched_records(match_df: pd.DataFrame, threshold: int = 80) -> pd.DataFrame:
    """
    Find records that did not match at the given threshold.
    
    Args:
        match_df: DataFrame with match results
        threshold: Score threshold for matches
        
    Returns:
        DataFrame with unmatched records
    """
    # Check if DataFrame is empty or missing match_score column
    if match_df.empty:
        logger.warning("Empty DataFrame provided to find_unmatched_records")
        return pd.DataFrame()
        
    if "match_score" not in match_df.columns:
        logger.warning("match_score column missing from DataFrame in find_unmatched_records")
        return match_df.copy()  # Return the whole dataframe as unmatched
    
    # Filter for records below threshold
    unmatched = match_df[match_df["match_score"] < threshold].copy()
    
    # Sort by match score if we have any records
    if not unmatched.empty and "match_score" in unmatched.columns:
        return unmatched.sort_values(by="match_score", ascending=False)
    
    return unmatched


def find_fjc_unmatched_records(match_df: pd.DataFrame, fjc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find FJC records that were not matched to any Congress records.
    
    Args:
        match_df: DataFrame with match results
        fjc_df: Original FJC DataFrame
        
    Returns:
        DataFrame with unmatched FJC records
    """
    # Check if either DataFrame is empty
    if fjc_df.empty:
        logger.warning("Empty FJC DataFrame provided to find_fjc_unmatched_records")
        return pd.DataFrame()
    
    if match_df.empty:
        logger.warning("Empty match DataFrame provided to find_fjc_unmatched_records; returning all FJC records as unmatched")
        return fjc_df.copy()
    
    # Check if nid column exists in both DataFrames
    if "nid" not in fjc_df.columns:
        logger.warning("nid column missing from FJC DataFrame in find_fjc_unmatched_records")
        return fjc_df.copy()  # Can't determine matches, so all are unmatched
        
    # Get NIDs that were matched
    if "nid" not in match_df.columns:
        logger.warning("nid column missing from match DataFrame in find_fjc_unmatched_records; returning all FJC records as unmatched")
        return fjc_df.copy()
    
    # Extract matched NIDs, handling potential NaN values
    matched_nids = set(match_df[~match_df["nid"].isna()]["nid"].unique())
    
    # If no matches were found, all FJC records are unmatched
    if not matched_nids:
        return fjc_df.copy()
    
    # Find FJC records not in the matched set
    unmatched_fjc = fjc_df[~fjc_df["nid"].isin(matched_nids)].copy()
    
    return unmatched_fjc


def generate_matching_summary(
    match_df: pd.DataFrame,
    fjc_df: pd.DataFrame,
    threshold: int = int(MATCH_THRESHOLD*100), # multiplied because displaying as percentile
    display_limit: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Generate summary information about the matching process.
    
    Args:
        match_df: DataFrame with match results
        fjc_df: Original FJC DataFrame
        threshold: Score threshold for matches
        display_limit: Maximum number of rows to include in each result category
        
    Returns:
        Dictionary with summary DataFrames
    """
    # Check if input DataFrames are empty
    if match_df.empty or fjc_df.empty:
        logger.warning("Empty DataFrame provided to generate_matching_summary")
        empty_df = pd.DataFrame()
        empty_dict = {}
        return {
            "high_confidence": empty_df,
            "high_confidence_sample": empty_df,
            "medium_confidence": empty_df,
            "medium_confidence_sample": empty_df,
            "low_confidence": empty_df if match_df.empty else match_df.copy(),
            "low_confidence_sample": empty_df if match_df.empty else match_df.head(display_limit),
            "ambiguous_matches": empty_df,
            "ambiguous_matches_sample": empty_df,
            "cong_unmatched": empty_df if match_df.empty else match_df.copy(),
            "cong_unmatched_sample": empty_df if match_df.empty else match_df.head(display_limit),
            "fjc_unmatched": empty_df if fjc_df.empty else fjc_df.copy(),
            "fjc_unmatched_sample": empty_df if fjc_df.empty else fjc_df.head(display_limit),
            "unmatched_with_reasons": empty_df,
            "reason_summary": empty_dict,
            "stats": {
                "total_cong_records": 0,
                "high_confidence_matches": 0,
                "medium_confidence_matches": 0,
                "low_confidence_matches": 0,
                "ambiguous_matches": 0,
                "total_fjc_records": 0,
                "unmatched_fjc_records": 0
            }
        }
    
    # Get categorized matches
    categorized = categorize_matches(match_df, threshold)
    
    # Filter out already matched records (rows with non-null nid values)
    # Only look for ambiguity in records that haven't been definitively matched yet
    unmatched_congress_df = match_df[match_df['nid'].isna()].copy() if 'nid' in match_df.columns else match_df.copy()
    
    # Only consider FJC records that have not already been matched
    # This is a significant optimization to avoid reprocessing all FJC records
    if 'nid' in match_df.columns and 'nid' in fjc_df.columns:
        # Get the NIDs of FJC records that have already been matched
        matched_nids = match_df[match_df['nid'].notna()]['nid'].unique()
        
        # Filter out FJC records that have already been matched
        relevant_fjc_df = fjc_df[~fjc_df['nid'].isin(matched_nids)].copy() if matched_nids.size > 0 else fjc_df.copy()
        
        logger.info(f"Excluded {len(matched_nids)} already matched FJC records from ambiguity check")
    else:
        # If we can't filter by NID, just use the whole FJC dataframe
        relevant_fjc_df = fjc_df.copy()
    
    logger.info(f"Processing {len(unmatched_congress_df)} unmatched records instead of {len(match_df)} total records")
    logger.info(f"Using {len(relevant_fjc_df)} relevant FJC records instead of {len(fjc_df)} total records")
    
    # Find ambiguous matches only in the unmatched records
    ambiguous = find_ambiguous_matches(unmatched_congress_df, relevant_fjc_df, threshold)
    
    # Find unmatched Congress records
    cong_unmatched = find_unmatched_records(match_df, threshold)
    
    # Find unmatched FJC records
    fjc_unmatched = find_fjc_unmatched_records(match_df, fjc_df)
    
    # Analyze reasons for match failures - handle potential errors
    try:
        unmatched_with_reasons, reason_summary, _ = analyze_match_failures(cong_unmatched, threshold)
    except Exception as e:
        logger.error(f"Error analyzing match failures: {e}")
        unmatched_with_reasons = cong_unmatched.copy() if not cong_unmatched.empty else pd.DataFrame()
        reason_summary = {}
    
    # Safely trim DataFrames to display limit
    high_conf_display = categorized["high_confidence"].head(display_limit) if not categorized["high_confidence"].empty else pd.DataFrame()
    med_conf_display = categorized["medium_confidence"].head(display_limit) if not categorized["medium_confidence"].empty else pd.DataFrame()
    low_conf_display = categorized["low_confidence"].head(display_limit) if not categorized["low_confidence"].empty else pd.DataFrame()
    ambiguous_display = ambiguous.head(display_limit) if not ambiguous.empty else pd.DataFrame()
    cong_unmatched_display = cong_unmatched.head(display_limit) if not cong_unmatched.empty else pd.DataFrame()
    fjc_unmatched_display = fjc_unmatched.head(display_limit) if not fjc_unmatched.empty else pd.DataFrame()
    
    # Create full result set
    result = {
        "high_confidence": categorized["high_confidence"],
        "high_confidence_sample": high_conf_display,
        "medium_confidence": categorized["medium_confidence"],
        "medium_confidence_sample": med_conf_display,
        "low_confidence": categorized["low_confidence"],
        "low_confidence_sample": low_conf_display,
        "ambiguous_matches": ambiguous,
        "ambiguous_matches_sample": ambiguous_display,
        "cong_unmatched": cong_unmatched,
        "cong_unmatched_sample": cong_unmatched_display,
        "fjc_unmatched": fjc_unmatched,
        "fjc_unmatched_sample": fjc_unmatched_display,
        "unmatched_with_reasons": unmatched_with_reasons,
        "reason_summary": reason_summary,
        "stats": {
            "total_cong_records": len(match_df),
            "high_confidence_matches": len(categorized["high_confidence"]),
            "medium_confidence_matches": len(categorized["medium_confidence"]),
            "low_confidence_matches": len(categorized["low_confidence"]),
            "ambiguous_matches": len(ambiguous),
            "total_fjc_records": len(fjc_df),
            "unmatched_fjc_records": len(fjc_unmatched)
        }
    }
    
    return result


def update_dataframe_with_matches(
    cong_df: pd.DataFrame,
    match_results: pd.DataFrame,
    threshold: int = int(MATCH_THRESHOLD*100),
    column_name: str = "fjc_nid"
) -> pd.DataFrame:
    """
    Update Congress DataFrame with matching FJC NIDs.
    
    Args:
        cong_df: Original Congress DataFrame
        match_results: DataFrame with match results
        threshold: Score threshold for high confidence matches
        column_name: Name for the column to add with matching NIDs
        
    Returns:
        Updated DataFrame with new column containing FJC NIDs
    """
    result_df = cong_df.copy()
    
    # Create a mapping of citation to NID for high confidence matches
    high_conf_matches = match_results[match_results["match_score"] >= threshold]
    
    if "citation" not in high_conf_matches.columns:
        logger.warning("Citation column not found in match results")
        return result_df
        
    if "nid" not in high_conf_matches.columns:
        logger.warning("NID column not found in match results")
        return result_df
    
    # Create mapping dictionary
    citation_to_nid = dict(zip(high_conf_matches["citation"], high_conf_matches["nid"]))
    
    # Add column with NID values
    result_df[column_name] = result_df["citation"].map(citation_to_nid)
    
    return result_df
