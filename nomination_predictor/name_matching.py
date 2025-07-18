from datetime import datetime
import re
from typing import Optional

from loguru import logger
from nameparser import HumanName
import pandas as pd


def normalize_text(text) -> str:
    """
    Normalize text by selectively replacing Spanish diacritical marks with their non-accented equivalents.
    Does not affect numbers or other non-letter characters (to allow this to still be usable on longer strings containing dates).

    Handles non-string inputs by converting to string or returning the input unchanged for non-string, non-numeric types.

    Examples:
        'Núñez' -> 'nunez'
        'Peña' -> 'pena'
        'López' -> 'lopez'
        'José Martínez Jr.' -> 'jose martinez jr.'
        123.45 -> '123.45'
        None -> ''

    Args:
        text: Input text to normalize (can be string, numeric, or other)

    Returns:
        Normalized text with Spanish accents removed and lowercased,
        or string version of numeric input, or original input for incompatible types
    """
    # Handle empty values
    if text is None or text == "":
        return ""

    # Handle non-string inputs
    if not isinstance(text, str):
        # Handle numeric types by converting to string
        if isinstance(text, (int, float)):
            try:
                return str(text).strip()
            except Exception as e:
                logger.warning(f"Failed to convert numeric value {repr(text)} to string: {e}")
                return ""
        else:
            # For other non-string types, log warning and return as-is
            logger.warning(
                f"normalize_text received non-string, non-numeric input: {type(text).__name__}"
            )
            return str(text) if hasattr(text, "__str__") else ""

    # Spanish specific replacements
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
        "Á": "A",
        "É": "E",
        "Í": "I",
        "Ó": "O",
        "Ú": "U",
        "Ü": "U",
        "Ñ": "N",
    }

    # Apply specific replacements
    normalized = text
    for accented, unaccented in replacements.items():
        normalized = normalized.replace(accented, unaccented)

    return normalized.lower().strip()


def split_names(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """
    Add three lowercase columns: first, last, middle_initial.
    Handles accent normalization for Spanish names and other special characters.

    Args:
        df: DataFrame containing names to parse
        name_col: Column name containing full names

    Returns:
        DataFrame with added first, last, and mi (middle initial) columns
    """
    parsed = df[name_col].fillna("").apply(HumanName)

    # Apply normalization to handle accented characters (e.g. ñ->n, á->a)
    df["first"] = parsed.apply(lambda n: normalize_text(n.first))
    df["last"] = parsed.apply(lambda n: normalize_text(n.last))
    df["mi"] = parsed.apply(lambda n: normalize_text(n.middle[:1] or ""))

    return df


def match_congress_to_fjc_by_name_and_date(
    congress_df: pd.DataFrame,
    fjc_judges_df: pd.DataFrame,
    temporal_window_days: int = 730,  # ±2 years
    congress_name_col: str = "nominee_name",
    congress_date_col: str = "receiveddate",
    nid_col: str = "nid",
) -> pd.DataFrame:
    """
    Match Congress nominations to FJC judges using exact name matching + date proximity.
    
    Strategy:
    1. Exact match on (last_name, first_name) between Congress and FJC judges
    2. For multiple matches, use closest nomination date as disambiguator
    3. Apply temporal sanity check (reject matches beyond window)
    4. Return Congress DataFrame with added 'nid' column
    
    Args:
        congress_df: DataFrame with Congress nomination data
        fjc_judges_df: DataFrame with FJC judge biographical data
        temporal_window_days: Maximum days between Congress and FJC nomination dates
        congress_name_col: Column containing Congress nominee names
        congress_date_col: Column containing Congress nomination dates
        nid_col: Column containing unique FJC identifier
        
    Returns:
        DataFrame with Congress records and matched NIDs
    """
    logger.info(
        f"Starting name+date matching with {len(congress_df)} Congress records and {len(fjc_judges_df)} FJC judges"
    )

    # Step 1: Parse names in both dataframes
    cong = split_names(congress_df.copy(), congress_name_col)
    fjc = split_names(fjc_judges_df.copy(), "last_name")  # FJC uses different column structure
    
    # Handle FJC name structure (last_name, first_name, middle_name columns)
    fjc["first"] = fjc_judges_df["first_name"].fillna("").apply(normalize_text)
    fjc["last"] = fjc_judges_df["last_name"].fillna("").apply(normalize_text)
    fjc["mi"] = fjc_judges_df["middle_name"].fillna("").apply(lambda x: normalize_text(x[:1]) if x else "")

    # Ensure we have a unique identifier for congress records
    if "congress_index" not in cong.columns:
        cong["congress_index"] = cong.index

    # Step 2: Exact match on last name and first name
    logger.info("Performing exact name matching on last and first name")
    name_matches = cong.merge(
        fjc[["first", "last", "mi", nid_col]], 
        on=["last", "first"], 
        how="inner", 
        suffixes=("", "_fjc")
    )
    
    logger.info(f"Found {len(name_matches)} name-based matches")
    
    if name_matches.empty:
        logger.warning("No exact name matches found between Congress and FJC data")
        # Return original congress_df with empty nid column
        result = congress_df.copy()
        result[nid_col] = pd.NA
        result["match_confidence"] = "no_match"
        return result

    # Step 3: Apply temporal disambiguation for multiple matches
    logger.info("Applying temporal disambiguation")
    
    # Check for ambiguous matches (multiple FJC judges for one Congress record)
    match_counts = name_matches.groupby("congress_index").size()
    ambiguous_indices = match_counts[match_counts > 1].index
    
    unambiguous_matches = name_matches[~name_matches["congress_index"].isin(ambiguous_indices)].copy()
    ambiguous_matches = name_matches[name_matches["congress_index"].isin(ambiguous_indices)].copy()
    
    logger.info(f"Found {len(unambiguous_matches)} unambiguous matches")
    logger.info(f"Found {len(ambiguous_matches)} ambiguous matches requiring date disambiguation")

    # Step 4: Resolve ambiguous matches using date proximity
    if not ambiguous_matches.empty:
        resolved_by_date = _resolve_matches_by_date_proximity(
            ambiguous_matches, fjc_judges_df, congress_date_col, temporal_window_days, nid_col
        )
        logger.info(f"Resolved {len(resolved_by_date)} matches using date proximity")
    else:
        resolved_by_date = pd.DataFrame()
    
    # Step 5: Try middle initial disambiguation for remaining ambiguous matches
    if not ambiguous_matches.empty:
        mi_resolved = ambiguous_matches.merge(
            fjc[["first", "last", "mi", nid_col]], 
            on=["last", "first", "mi"], 
            how="inner"
        ).drop_duplicates(subset=["congress_index"])
        
        # Remove matches already resolved by date
        if not resolved_by_date.empty:
            mi_resolved = mi_resolved[~mi_resolved["congress_index"].isin(resolved_by_date["congress_index"])]
        
        logger.info(f"Resolved {len(mi_resolved)} additional matches using middle initial")
    else:
        mi_resolved = pd.DataFrame()

    # Step 6: Combine all successful matches
    all_matches = []
    
    # Add unambiguous matches
    if not unambiguous_matches.empty:
        unambiguous_matches["match_confidence"] = "high_unambiguous"
        all_matches.append(unambiguous_matches)
    
    # Add date-resolved matches
    if not resolved_by_date.empty:
        resolved_by_date["match_confidence"] = "high_date_resolved"
        all_matches.append(resolved_by_date)
    
    # Add middle-initial resolved matches
    if not mi_resolved.empty:
        mi_resolved["match_confidence"] = "medium_mi_resolved"
        all_matches.append(mi_resolved)
    
    # Combine all matches
    if all_matches:
        final_matches = pd.concat(all_matches, ignore_index=True)
        # Keep only the best match per congress record
        final_matches = final_matches.drop_duplicates(subset=["congress_index"], keep="first")
    else:
        final_matches = pd.DataFrame(columns=["congress_index", nid_col, "match_confidence"])
    
    # Step 7: Merge back to original Congress DataFrame
    result = congress_df.copy()
    if "congress_index" not in result.columns:
        result["congress_index"] = result.index
    
    # Merge in the NIDs
    result = result.merge(
        final_matches[["congress_index", nid_col, "match_confidence"]], 
        on="congress_index", 
        how="left"
    )
    
    # Fill unmatched records
    result[nid_col] = result[nid_col].fillna(pd.NA)
    result["match_confidence"] = result["match_confidence"].fillna("no_match")
    
    # Clean up temporary column
    result = result.drop(columns=["congress_index"], errors="ignore")
    
    # Log summary statistics
    total_congress = len(result)
    matched_count = result[nid_col].notna().sum()
    
    logger.info(f"Matching complete: {matched_count}/{total_congress} Congress records matched")
    logger.info("Match confidence distribution:")
    logger.info(result["match_confidence"].value_counts())
    
    return result


def _resolve_matches_by_date_proximity(
    ambiguous_matches: pd.DataFrame,
    fjc_judges_df: pd.DataFrame,
    congress_date_col: str,
    temporal_window_days: int,
    nid_col: str,
) -> pd.DataFrame:
    """
    Resolve ambiguous name matches using nomination date proximity.
    
    For each Congress record with multiple FJC matches, find the FJC judge
    whose nomination date is closest to the Congress nomination date.
    
    Args:
        ambiguous_matches: DataFrame with multiple FJC matches per Congress record
        fjc_judges_df: Full FJC judges DataFrame with nomination dates
        congress_date_col: Column name for Congress nomination dates
        temporal_window_days: Maximum allowed days between nomination dates
        nid_col: Column name for NID
        
    Returns:
        DataFrame with resolved matches (one per Congress record)
    """
    if ambiguous_matches.empty:
        return pd.DataFrame()
    
    resolved_matches = []
    
    # Get nomination date columns from FJC data
    nom_date_cols = [f"nomination_date_({i})" for i in range(1, 7)]
    available_nom_cols = [col for col in nom_date_cols if col in fjc_judges_df.columns]
    
    if not available_nom_cols:
        logger.warning("No nomination date columns found in FJC data")
        return pd.DataFrame()
    
    # Group by congress_index to resolve each ambiguous case
    for congress_idx, group in ambiguous_matches.groupby("congress_index"):
        congress_date = group[congress_date_col].iloc[0]
        
        if pd.isna(congress_date):
            continue  # Skip if no Congress date available
        
        congress_dt = pd.to_datetime(congress_date, errors="coerce")
        if pd.isna(congress_dt):
            continue  # Skip if date parsing failed
        
        best_match = None
        min_days_diff = float('inf')
        
        # Check each potential FJC match
        for _, match_row in group.iterrows():
            nid = match_row[nid_col]
            
            # Get FJC judge's nomination dates
            fjc_judge = fjc_judges_df[fjc_judges_df[nid_col] == nid]
            if fjc_judge.empty:
                continue
            
            fjc_judge = fjc_judge.iloc[0]  # Take first row if multiple
            
            # Check all nomination date columns for this judge
            for nom_col in available_nom_cols:
                fjc_date = fjc_judge[nom_col]
                
                if pd.isna(fjc_date):
                    continue
                
                fjc_dt = pd.to_datetime(fjc_date, errors="coerce")
                if pd.isna(fjc_dt):
                    continue
                
                # Calculate days difference
                days_diff = abs((congress_dt - fjc_dt).days)
                
                # Check if within temporal window and better than current best
                if days_diff <= temporal_window_days and days_diff < min_days_diff:
                    min_days_diff = days_diff
                    best_match = match_row.copy()
                    best_match["days_diff"] = days_diff
                    best_match["fjc_nomination_date"] = fjc_date
        
        # Add best match if found
        if best_match is not None:
            resolved_matches.append(best_match)
    
    if resolved_matches:
        return pd.DataFrame(resolved_matches)
    else:
        return pd.DataFrame()


def _apply_temporal_sanity_check(
    matches_df: pd.DataFrame,
    congress_df: pd.DataFrame, 
    fjc_judges_df: pd.DataFrame,
    nid_col: str,
) -> pd.DataFrame:
    """
    Filter out matches where FJC judge died before the nomination date.
    
    Args:
        matches_df: DataFrame with name matches
        congress_df: Original Congress DataFrame with receiveddate
        fjc_judges_df: FJC judges DataFrame with death_year
        nid_col: Column name for NID
        
    Returns:
        Filtered matches DataFrame
    """
    if matches_df.empty:
        return matches_df
    
    # Merge in receiveddate from congress_df
    matches_with_dates = matches_df.merge(
        congress_df[["congress_index", "receiveddate"]].drop_duplicates(),
        on="congress_index",
        how="left"
    )
    
    # Merge in death_year from fjc_judges_df
    matches_with_dates = matches_with_dates.merge(
        fjc_judges_df[[nid_col, "death_year"]].drop_duplicates(subset=[nid_col]),
        on=nid_col,
        how="left"
    )
    
    # Extract nomination year from receiveddate
    matches_with_dates["nomination_year"] = pd.to_datetime(
        matches_with_dates["receiveddate"], errors="coerce"
    ).dt.year
    
    # Filter out impossible matches (judge died before nomination)
    before_count = len(matches_with_dates)
    valid_mask = (
        matches_with_dates["death_year"].isna() |  # Judge still alive or death unknown
        (matches_with_dates["nomination_year"].isna()) |  # Nomination date unknown
        (matches_with_dates["death_year"] >= matches_with_dates["nomination_year"])  # Died after nomination
    )
    
    filtered_matches = matches_with_dates[valid_mask].copy()
    after_count = len(filtered_matches)
    
    if before_count > after_count:
        logger.info(f"Temporal sanity check removed {before_count - after_count} impossible matches")
    
    # Drop the temporary columns
    return filtered_matches.drop(columns=["receiveddate", "death_year", "nomination_year"], errors="ignore")


def _parse_other_nom_text(txt: str) -> dict:
    """
    Extract court, nomination_date, and failure_reason from an FJC
    'other_nominations/recess_appointments' string.
      • Accepts both 'Nominated to ...' and 'Received recess appointment to ...'
    Returns an empty dict if pattern not recognized.
    """
    if pd.isna(txt):
        return {}
    pattern = (
        r"^(?:Nominated to|Received recess appointment to)\s+"
        r"(?P<court>.*?),\s+"
        r"(?P<date>[A-Z][a-z]+\s\d{1,2},\s\d{4});\s+"
        r"(?P<reason>.+)$"
    )
    m = re.match(pattern, str(txt).strip())
    if not m:
        return {}
    try:
        nom_dt = datetime.strptime(m.group("date"), "%B %d, %Y").date()
    except ValueError:
        nom_dt = None
    return {
        "court_name_other": m.group("court"),
        "other_nom_date": nom_dt,
        "other_nom_failure_reason": m.group("reason").strip().lower(),
    }


def prep_fjc_other(fjc_other_df: pd.DataFrame) -> pd.DataFrame:
    """Add court / date / reason + split names (first/last/mi) for fast merge."""
    parsed = fjc_other_df["other_nominations/recess_appointments"].apply(_parse_other_nom_text)
    fjc_other_df = pd.concat([fjc_other_df, parsed.apply(pd.Series)], axis=1)
    fjc_other_df = split_names(fjc_other_df, "judge_name")  # ← from name_matching.py
    return fjc_other_df


# for parsing predecessor names from description
_VICE_RE = re.compile(
    r"""
    \bvice\s+                     # literal 'vice' and following spaces
    (?P<pre>[^,\.]+?)             # capture up to , or . (non-greedy)
    (?=[,\.]|$)                   # look-ahead stop at comma / period / end
    """,
    flags=re.I | re.VERBOSE,
)

# for parsing reason for nomination from description
_REASON_SPLIT_RE = re.compile(
    r"""
    \s*,\s*                 # the comma just before the reason
    (retir|elevat|deceas|withdraw|no\ senate\ vote|reject|resign)
    """,
    flags=re.I | re.VERBOSE,
)


def extract_vacancy_reason(desc: str) -> Optional[str]:
    """
    Extract vacancy reason from the description field.
    
    Takes a description string that might contain reason information
    after a judge's name (e.g., "vice John Smith, retiring") and extracts
    the standardized reason category.
    
    Args:
        desc: String containing description text
        
    Returns:
        Standardized vacancy reason or None if not found
        
    Example:
        "vice John Smith, retiring" -> "Retirement"
        "vice Jane Doe, elevated" -> "Appointment to Another Judicial Position"
    """
    if not isinstance(desc, str) or not desc:
        return None
        
    # First check if there's a predecessor mentioned
    m = _VICE_RE.search(desc)
    if not m:
        return None
        
    # Check if predecessor contains "new position" phrase
    predecessor = m.group("pre").lower()
    if "new position" in predecessor:
        return "New Position"
        
    # Look for a reason after the predecessor's name
    reason_match = _REASON_SPLIT_RE.search(desc)
    if not reason_match:
        return None
        
    # Map the raw reason text to standardized categories
    reason_text = reason_match.group(1).lower()
    
    # Map variations to standardized categories
    if "retir" in reason_text:
        return "Retirement"
    elif "elevat" in reason_text:
        return "Elevated"
    elif "deceas" in reason_text:
        return "Death"
    elif "resign" in reason_text:
        return "Resignation"
    elif "withdr" in reason_text:
        return "Withdrawal"
    elif "reject" in reason_text:
        return "Rejection"
    elif "reappoint" in reason_text:
        return "Reappointment"
    
    # If we detected a reason pattern but it doesn't match our categories,
    # return the raw text for further analysis
    return reason_text.capitalize()


def fill_vacancy_reason_column(df: pd.DataFrame,
                               desc_col: str = "description",
                               target_col: str = "vacancy_reason"
                              ) -> pd.DataFrame:
    """
    Populate missing values in `target_col` by parsing vacancy reasons from `desc_col`.
    
    Args:
        df: DataFrame containing the data
        desc_col: Name of column containing description text
        target_col: Name of column to populate with vacancy reasons
        
    Returns:
        DataFrame with vacancy_reason column filled
    """
    # Create the column if it doesn't exist
    if target_col not in df.columns:
        df[target_col] = None
    
    # Only process rows where target column is missing and description is available
    mask = df[target_col].isna() & df[desc_col].notna()
    
    # Apply the extraction function
    df.loc[mask, target_col] = df.loc[mask, desc_col].apply(extract_vacancy_reason)
    
    # Log the number of filled values
    num_filled = mask.sum()
    logger.info(f"Extracted {num_filled} vacancy reasons from descriptions")
    
    return df

_PUBLIC_LAW_RE = re.compile(
    r"p\.\s*l\.?", flags=re.I
)  # p. l. / p.l.  # for parsing a thing typically inserted into descriptions about new positions


# ------------------------------------------------------------------
def extract_predecessor(desc: str) -> Optional[str]:
    """
    Parse `desc` (Congress nomination description) and return the predecessor
    string after the word “vice …”, or the normalized “new position …” phrase.
    Returns `None` when no predecessor phrase is found.

    Handles all examples:
      • commas / periods around suffixes (",jr.," etc.)
      • various 'new position created by p. l. / P.L.' strings
      • ignores case
    """
    if not isinstance(desc, str):
        return None

    #  locate the word "vice" (case-insensitive)
    m = re.search(r"\bvice\b", desc, flags=re.I)
    if not m:
        return None

    # substring that follows 'vice'
    after = desc[m.end() :].lstrip()

    # 2split off the trailing reason (retired / elevated / ...)
    m_reason = _REASON_SPLIT_RE.search(after)
    if m_reason:
        name_part = after[: m_reason.start()]
    else:
        # If no comma+reason, take the rest of the string
        name_part = after

    name_part = name_part.strip(" ,.")

    if not name_part:
        return None

    #   normalize 'new position'
    if name_part.lower().startswith("a new position"):
        name_part = _PUBLIC_LAW_RE.sub("public law", name_part.lower()).strip()
        name_part = re.sub(r"\s+", " ", name_part)  # collapse spaces
        # capitalize 'public law' phrase for readability
        name_part = (
            name_part.replace("public law", "public law").replace(
                "public law", "public law"
            )  # idempotent
        )

    return name_part


# ------------------------------------------------------------------
# 2.  dataframe patcher
# ------------------------------------------------------------------
def fill_predecessor_column(
    df: pd.DataFrame, desc_col: str = "description", target_col: str = "nominees_0_predecessorname"
) -> pd.DataFrame:
    """
    Populate missing values in `target_col` by parsing `desc_col`.
    Returns the DataFrame with the column filled in-place.
    """
    mask = df[target_col].isna() | (df[target_col].str.strip() == "")
    df.loc[mask, target_col] = df.loc[mask, desc_col].apply(extract_predecessor)
    return df
