from datetime import date
import re
from typing import Dict, Tuple

from loguru import logger
import pandas as pd
from thefuzz import fuzz
from tqdm import tqdm

# ---------------------------------------------------------------------------
# FUZZY MATCHING PIPELINE
# ---------------------------------------------------------------------------


def normalize_date_format(date_str):
    """
    Convert various date string formats to a standardized datetime object.

    Args:
        date_str: Date string in various formats or None/NaN

    Returns:
        datetime.date object or None if conversion fails
    """
    if date_str is None or pd.isna(date_str):
        return None

    # If already a date object, return as is
    if isinstance(date_str, date):
        return date_str

    try:
        return pd.to_datetime(date_str).date()
    except (ValueError, TypeError):
        logger.debug(f"Could not parse date: {date_str}")
        return None


def calculate_name_similarity(name1: str, name2: str) -> int:
    """
    Calculate similarity between two judge names using fuzzy matching.

    Args:
        name1: First name string
        name2: Second name string

    Returns:
        Similarity score from 0-100
    """
    if not name1 or not name2 or pd.isna(name1) or pd.isna(name2):
        return 0

    # Normalize names: lowercase, strip extra whitespace
    name1 = str(name1).lower().strip()
    name2 = str(name2).lower().strip()

    # Handle last name, first name format
    if "," in name1:
        parts = name1.split(",", 1)
        name1 = f"{parts[1].strip()} {parts[0].strip()}"

    if "," in name2:
        parts = name2.split(",", 1)
        name2 = f"{parts[1].strip()} {parts[0].strip()}"

    # Calculate token set ratio which handles word order and partial matches well
    return fuzz.token_set_ratio(name1, name2)


def calculate_court_similarity(court1: str, court2: str) -> int:
    """
    Calculate similarity between two court names using fuzzy matching.

    Args:
        court1: First court name string
        court2: Second court name string

    Returns:
        Similarity score from 0-100
    """
    if not court1 or not court2 or pd.isna(court1) or pd.isna(court2):
        return 0

    # Normalize court names: lowercase, strip extra whitespace
    court1 = str(court1).lower().strip()
    court2 = str(court2).lower().strip()

    # Standardize some common variations
    court1 = court1.replace("united states", "u.s.")
    court2 = court2.replace("united states", "u.s.")

    # Calculate token set ratio which handles word order and partial matches well
    return fuzz.token_set_ratio(court1, court2)


def calculate_date_similarity(date1: str, date2: str, max_days_diff: int = 45) -> int:
    """
    Calculate similarity between two dates based on their proximity.

    Args:
        date1: First date string
        date2: Second date string
        max_days_diff: Maximum days difference for non-zero similarity (default: 45)

    Returns:
        Similarity score from 0-100
    """
    # Convert to datetime.date objects
    date1_obj = normalize_date_format(date1)
    date2_obj = normalize_date_format(date2)

    if not date1_obj or not date2_obj:
        return 0

    # Calculate days difference
    days_diff = abs((date1_obj - date2_obj).days)

    # If difference exceeds max window, return 0
    if days_diff > max_days_diff:
        return 0

    # Calculate similarity score (100 for same date, decreasing as days_diff increases)
    return int(100 * (1 - days_diff / max_days_diff))


def find_matches_with_blocking(
    congress_df: pd.DataFrame,
    fjc_df: pd.DataFrame,
    threshold: int = 80,
    name_weight: float = 0.5,
    court_weight: float = 0.3,
    date_weight: float = 0.2,
    blocking_column: str = "last_name",
) -> pd.DataFrame:
    """
    Find matches between Congress.gov nominations and FJC judge data using
    a blocking approach followed by weighted fuzzy matching.

    Args:
        congress_df: DataFrame with Congress.gov nomination data
        fjc_df: DataFrame with FJC judge data
        threshold: Minimum score to consider a match (0-100)
        name_weight: Weight for name similarity in final score
        court_weight: Weight for court similarity in final score
        date_weight: Weight for date similarity in final score
        blocking_column: Column to use for initial blocking/filtering

    Returns:
        DataFrame with match results, including original columns and match scores
    """
    if not name_weight + court_weight + date_weight == 1.0:
        logger.warning("Weights do not sum to 1.0, normalizing automatically")
        total = name_weight + court_weight + date_weight
        name_weight /= total
        court_weight /= total
        date_weight /= total

    logger.info(
        f"Starting fuzzy matching with {len(congress_df)} Congress records and {len(fjc_df)} FJC records"
    )

    # Create result DataFrame to store matches
    results = []

    # Determine blocking column to use based on what's available in the dataframes
    cong_blocking_col = blocking_column
    if (
        blocking_column not in congress_df.columns
        and "last_name_from_full_name" in congress_df.columns
    ):
        cong_blocking_col = "last_name_from_full_name"

    # Process each Congress record
    for idx, congress_row in tqdm(
        congress_df.iterrows(), total=len(congress_df), desc="Matching records"
    ):
        best_match = None
        best_score = 0
        congress_last_name = congress_row.get(cong_blocking_col, "")

        # Skip if no blocking value
        if not congress_last_name or pd.isna(congress_last_name):
            logger.debug(f"Skipping Congress record {idx} - no blocking value")
            # Still add to results with zero match score
            result_row = {col: congress_row.get(col) for col in congress_df.columns}
            result_row["match_score"] = 0
            result_row["nid"] = None
            results.append(result_row)
            continue

        # Filter FJC records by blocking column for efficiency
        # Handle case where column names might be different in the two dataframes
        try:
            potential_matches = fjc_df[
                fjc_df[blocking_column].str.lower() == congress_last_name.lower()
            ]

            if len(potential_matches) == 0:
                logger.debug(f"No potential matches found for {congress_last_name}")
        except (KeyError, AttributeError):
            logger.warning(f"Blocking column '{blocking_column}' not found in FJC data")
            potential_matches = fjc_df  # Fall back to full dataset if column missing

        # For each potential match, calculate similarity
        for fjc_idx, fjc_row in potential_matches.iterrows():
            # Calculate component similarities
            name_sim = calculate_name_similarity(
                congress_row.get("full_name", ""), fjc_row.get("name", "")
            )

            court_sim = calculate_court_similarity(
                congress_row.get("court_name", ""), fjc_row.get("court", "")
            )

            date_sim = calculate_date_similarity(
                congress_row.get("receiveddate", ""), fjc_row.get("nomination_date", "")
            )

            # Calculate weighted score
            score = name_sim * name_weight + court_sim * court_weight + date_sim * date_weight

            # Update best match if score is higher
            if score > best_score:
                best_score = score
                best_match = fjc_row

        # Create result row with match information
        result_row = {}

        # Add Congress data
        for col in congress_df.columns:
            result_row[col] = congress_row.get(col)

        # Add match information
        if best_match is not None and best_score >= threshold:
            result_row["match_score"] = best_score
            result_row["nid"] = best_match.get("nid")
            # Add any other FJC columns you want to include
        else:
            result_row["match_score"] = best_score if best_score > 0 else 0
            result_row["nid"] = None

        results.append(result_row)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Handle the case where results is empty
    if results_df.empty:
        logger.info("No matches found (empty results DataFrame)")
        # Ensure the DataFrame has the necessary columns even when empty
        # First create with the match_score column
        results_df = pd.DataFrame(columns=["match_score", "nid"] + list(congress_df.columns))
        return results_df

    # Ensure match_score column exists
    if "match_score" not in results_df.columns:
        results_df["match_score"] = 0
        matched_count = 0
    else:
        matched_count = len(results_df[results_df["match_score"] >= threshold])

    logger.info(
        f"Matching complete. Found {matched_count} matches out of {len(results_df)} records"
    )

    return results_df


def create_blocking_key(row, last_name_col, date_col=None, court_col=None):
    """
    Create a blocking key to reduce the number of comparisons needed for fuzzy matching.
    Uses a combination of last name initial and optionally court/date for efficient blocking.

    Args:
        row: DataFrame row with nominee information
        last_name_col: Column name containing the last name
        date_col: Optional column name containing a date (e.g., nomination date)
        court_col: Optional column name containing court information

    Returns:
        A tuple that can be used as a blocking key
    """
    components = []

    # Always use the first letter of the last name if available
    if last_name_col in row and row[last_name_col]:
        last_name = str(row[last_name_col]).strip().lower()
        if last_name:
            components.append(last_name[0])

    # Add court region/type if available
    if court_col and court_col in row and row[court_col]:
        court = str(row[court_col]).lower()
        # Extract key court tokens (circuit/district/state) for blocking
        court_tokens = set()
        if "circuit" in court:
            court_tokens.add("circuit")
        if "district" in court:
            court_tokens.add("district")
        # Add any state codes found
        for state_pattern in [r"\b[a-z]{2}\b", r"\b(north|south|east|west)\b"]:
            state_matches = re.findall(state_pattern, court)
            court_tokens.update(state_matches)

        components.extend(sorted(court_tokens))

    # Add year from date if available
    if date_col and date_col in row and row[date_col]:
        date_obj = normalize_date_format(row[date_col])
        if date_obj:
            components.append(str(date_obj.year))

    # Return tuple for efficient dict lookup
    return tuple(components)


def calculate_match_score(cong_row, fjc_row, name_weight=0.6, court_weight=0.25, date_weight=0.15):
    """
    Calculate a weighted match score between a Congress.gov nomination and FJC judge record.

    Args:
        cong_row: Row from Congress.gov nominations DataFrame
        fjc_row: Row from FJC judges DataFrame
        name_weight: Weight to assign to name similarity (default: 0.6)
        court_weight: Weight to assign to court similarity (default: 0.25)
        date_weight: Weight to assign to date similarity (default: 0.15)

    Returns:
        Float score between 0-100 representing overall match confidence
    """
    # Calculate name similarity (most important factor)
    name_sim = calculate_name_similarity(
        cong_row.get("full_name") or cong_row.get("full_name_from_description"),
        fjc_row.get("name") or fjc_row.get("full_name"),
    )

    # Calculate court similarity
    court_sim = calculate_court_similarity(
        cong_row.get("court_name") or cong_row.get("nominees_0_positiontitle"),
        fjc_row.get("court"),
    )

    # Calculate date similarity
    date_sim = calculate_date_similarity(
        cong_row.get("receiveddate"), fjc_row.get("nomination_date")
    )

    # Calculate weighted score
    weighted_score = (
        (name_sim * name_weight) + (court_sim * court_weight) + (date_sim * date_weight)
    )

    return weighted_score


def analyze_match_failures(
    nominees_df: pd.DataFrame, threshold: int = 80
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
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
    unmatched = (
        nominees_df[nominees_df["match_score"] < threshold].copy()
        if "match_score" in nominees_df.columns
        else nominees_df.copy()
    )

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
            reasons.append(
                f"Marginal match (score {row['match_score']:.1f}) - check name and court"
            )

    unmatched["failure_reason"] = reasons

    # Count occurrences of each failure reason
    reason_counts = unmatched["failure_reason"].value_counts().reset_index()
    reason_counts.columns = ["Failure Reason", "Count"]

    # Get examples for each failure reason
    examples = {}
    for reason in reason_counts["Failure Reason"].unique():
        examples[reason] = unmatched[unmatched["failure_reason"] == reason].head(3)[
            ["full_name", "court_clean", "match_score", "failure_reason"]
        ]

    return unmatched, reason_counts, examples
