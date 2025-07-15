from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import re

from loguru import logger
import numpy as np
import pandas as pd

from nomination_predictor.config import INTERIM_DATA_DIR, RAW_DATA_DIR


def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")


def merge_latest_education(
    cong_df: pd.DataFrame,
    fjc_education_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each nomination pick *one* education row (latest degree_year ≤ receiveddate).
    Adds: degree, school, degree_year, education_sequence  (all from that row).
    """
    _ensure_datetime(cong_df, "receiveddate")
    edu = fjc_education_df.copy()
    edu["degree_year"] = pd.to_numeric(edu["degree_year"], errors="coerce")

    # Keep only rows whose degree_year ≤ nomination year
    merged = cong_df[["nid", "receiveddate"]].merge(edu, on="nid", how="left")
    merged = merged[
        merged["degree_year"].notna() & (merged["degree_year"] <= merged["receiveddate"].dt.year)
    ]

    # pick latest degree_year, then highest sequence if tie
    merged.sort_values(["nid", "degree_year", "sequence"], inplace=True)
    best = merged.groupby(["nid", "receiveddate"]).tail(1)

    best = best.rename(columns={"sequence": "education_sequence"})[
        ["nid", "receiveddate", "degree", "school", "degree_year", "education_sequence"]
    ]

    # left‑join back
    out = cong_df.merge(best, on=["nid", "receiveddate"], how="left")
    return out


def _start_year(text: str | float) -> float:
    """Return first 4‑digit year in string or np.nan."""
    if pd.isna(text):
        return np.nan
    m = pd.to_numeric(pd.Series(text).str.extract(r"(\d{4})")[0], errors="coerce")
    return m.iat[0]


def merge_latest_career(
    cong_df: pd.DataFrame,
    fjc_career_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Choose the career row with *max sequence* whose start_year ≤ nomination year.
    Adds: professional_career, professional_career_sequence
    """
    _ensure_datetime(cong_df, "receiveddate")
    career = fjc_career_df.copy()
    career["start_year"] = career["professional_career"].apply(_start_year)

    merged = cong_df[["nid", "receiveddate"]].merge(career, on="nid", how="left")
    merged = merged[
        merged["start_year"].notna() & (merged["start_year"] <= merged["receiveddate"].dt.year)
    ]

    merged.sort_values(["nid", "sequence"], inplace=True)
    best = merged.groupby(["nid", "receiveddate"]).tail(1)
    best = best.rename(columns={"sequence": "professional_career_sequence"})[
        ["nid", "receiveddate", "professional_career", "professional_career_sequence"]
    ]
    out = cong_df.merge(best, on=["nid", "receiveddate"], how="left")
    return out


def merge_nearest_fed_service(
    cong_df: pd.DataFrame,
    fjc_service_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pick the federal_service row whose nomination_date is *closest* (≤) to receiveddate.
    Adds all service columns for that single row, plus fed_service_sequence.
    """
    _ensure_datetime(cong_df, "receiveddate")
    service = fjc_service_df.copy()
    _ensure_datetime(service, "nomination_date")

    merged = cong_df[["nid", "receiveddate"]].merge(
        service, on="nid", how="left", suffixes=("", "_svc")
    )
    merged = merged[
        merged["nomination_date"].notna() & (merged["nomination_date"] <= merged["receiveddate"])
    ]

    merged["abs_diff"] = (merged["receiveddate"] - merged["nomination_date"]).abs()

    merged.sort_values(["nid", "receiveddate", "abs_diff", "sequence"], inplace=True)
    best = merged.groupby(["nid", "receiveddate"]).head(1)
    best = best.rename(columns={"sequence": "fed_service_sequence"})

    base_cols = ["nid", "receiveddate", "fed_service_sequence"]
    extra_cols = [
        c
        for c in best.columns
        if c not in ("abs_diff", "nid", "receiveddate") and not c.endswith("_svc")
    ]
    keep_cols = base_cols + extra_cols  # now unique

    out = cong_df.merge(best[keep_cols], on=["nid", "receiveddate"], how="left")
    return out



# ---------------------------------------------------------------------------
# 1.  PRESIDENTIAL TERMS WITH PARTY AFFILIATION AND FULL NAMES
#    tuple = (president_number, first_inauguration_date, last_day_in_office, party, full_name)
#    party: 'D' = Democratic, 'R' = Republican, 'DR' = Democratic-Republican,
#           'F' = Federalist, 'W' = Whig, 'U' = Union, etc.
#    full_name: First Middle Last format, for filling appointing_president column
# ---------------------------------------------------------------------------

PRESIDENCY_DATES = [
    # Format: (number, start_date, end_date, party, full_name)
    # Names formatted as First Middle Last to match appointing_president column format
    (47, date(2025, 1, 20), date(2029, 1, 20), "R", "Donald John Trump"),
    (46, date(2021, 1, 20), date(2025, 1, 20), "D", "Joseph Robinette Biden"),
    (45, date(2017, 1, 20), date(2021, 1, 20), "R", "Donald John Trump"),
    (44, date(2009, 1, 20), date(2017, 1, 20), "D", "Barack Hussein Obama"),
    (43, date(2001, 1, 20), date(2009, 1, 20), "R", "George Walker Bush"),
    (42, date(1993, 1, 20), date(2001, 1, 20), "D", "William Jefferson Clinton"),
    (41, date(1989, 1, 20), date(1993, 1, 20), "R", "George Herbert Walker Bush"),
    (40, date(1981, 1, 20), date(1989, 1, 20), "R", "Ronald Wilson Reagan"),
    (39, date(1977, 1, 20), date(1981, 1, 20), "D", "James Earl Carter"),
    (38, date(1974, 8, 9), date(1977, 1, 20), "R", "Gerald Rudolph Ford"),
    (37, date(1969, 1, 20), date(1974, 8, 9), "R", "Richard Milhous Nixon"),
    (36, date(1963, 11, 22), date(1969, 1, 20), "D", "Lyndon Baines Johnson"),
    (35, date(1961, 1, 20), date(1963, 11, 22), "D", "John Fitzgerald Kennedy"),
    (34, date(1953, 1, 20), date(1961, 1, 20), "R", "Dwight David Eisenhower"),
    (33, date(1945, 4, 12), date(1953, 1, 20), "D", "Harry S Truman"),
    (32, date(1933, 3, 4), date(1945, 4, 12), "D", "Franklin Delano Roosevelt"),
    (31, date(1929, 3, 4), date(1933, 3, 4), "R", "Herbert Clark Hoover"),
    (30, date(1923, 8, 2), date(1929, 3, 4), "R", "John Calvin Coolidge"),
    (29, date(1921, 3, 4), date(1923, 8, 2), "R", "Warren Gamaliel Harding"),
    (28, date(1913, 3, 4), date(1921, 3, 4), "D", "Thomas Woodrow Wilson"),
    (27, date(1909, 3, 4), date(1913, 3, 4), "R", "William Howard Taft"),
    (26, date(1901, 9, 14), date(1909, 3, 4), "R", "Theodore Roosevelt"),
    (25, date(1897, 3, 4), date(1901, 9, 14), "R", "William McKinley"),
    (24, date(1893, 3, 4), date(1897, 3, 4), "D", "Stephen Grover Cleveland"),  # 2nd term
    (23, date(1889, 3, 4), date(1893, 3, 4), "R", "Benjamin Harrison"),
    (22, date(1885, 3, 4), date(1889, 3, 4), "D", "Stephen Grover Cleveland"),  # 1st term
    (21, date(1881, 9, 19), date(1885, 3, 4), "R", "Chester Alan Arthur"),
    (
        20,
        date(
            1881,
            3,
            4,
        ),
        date(1881, 9, 19),
        "R",
        "James Abram Garfield",
    ),
    (19, date(1877, 3, 4), date(1881, 3, 4), "R", "Rutherford Birchard Hayes"),
    (18, date(1869, 3, 4), date(1877, 3, 4), "R", "Ulysses Simpson Grant"),
    (17, date(1865, 4, 15), date(1869, 3, 4), "D", "Andrew Johnson"),
    (16, date(1861, 3, 4), date(1865, 4, 15), "R", "Abraham Lincoln"),
    (15, date(1857, 3, 4), date(1861, 3, 4), "D", "James Buchanan"),
    (14, date(1853, 3, 4), date(1857, 3, 4), "D", "Franklin Pierce"),
    # If you're able to retrieve judicial & congressional data via API dating back further, congratulations on your data gathering!
    # Until then this is good enough for the data I've been able to obtain.
]

# Pre-sort newest-first so we can break quickly in lookups
PRESIDENCY_DATES.sort(key=lambda t: t[1], reverse=True)


# Example of filling in missing appointing president values
def fill_missing_appointing_presidents(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the appointing_president column using nomination date."""
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Check if appointing_president column is already fully populated
    if not result_df["appointing_president"].isna().any():
        logger.info(
            "No missing appointing president values detected - dataframe already fully populated"
        )
        return result_df

    # Only process rows where appointing_president is missing but nomination_date is available
    mask = result_df["appointing_president"].isna() & result_df["receiveddate"].notna()

    # Check if we have any rows to process
    if not mask.any():
        logger.info(
            "No rows available to fill: either no missing appointing president values or missing reception dates"
        )
        return result_df

    # Apply the president_name function to get the appointing president
    result_df.loc[mask, "appointing_president"] = result_df.loc[mask, "receiveddate"].apply(
        president_name
    )

    # Log the number of filled values
    num_filled = mask.sum()
    logger.info(f"Filled {num_filled} missing appointing president values using nomination dates")

    return result_df


def fill_missing_party_of_appointing_presidents(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the party_of_appointing_president column using nomination date."""
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Check if party_of_appointing_president column exists, create if not
    if "party_of_appointing_president" not in result_df.columns:
        result_df["party_of_appointing_president"] = None
        logger.info("Created party_of_appointing_president column which was missing")

    # Check if party_of_appointing_president column is already fully populated
    if not result_df["party_of_appointing_president"].isna().any():
        logger.info(
            "No missing party of appointing president values detected - dataframe already fully populated"
        )
        return result_df

    # Only process rows where party_of_appointing_president is missing but receiveddate is available
    mask = result_df["party_of_appointing_president"].isna() & result_df["receiveddate"].notna()

    # Check if we have any rows to process
    if not mask.any():
        logger.info(
            "No rows available to fill: either no missing party values or missing reception dates"
        )
        return result_df

    # Apply the president_party function to get the party of the appointing president
    result_df.loc[mask, "party_of_appointing_president"] = result_df.loc[
        mask, "receiveddate"
    ].apply(president_party)

    # Log the number of filled values
    num_filled = mask.sum()
    logger.info(
        f"Filled {num_filled} missing party of appointing president values using nomination dates"
    )

    return result_df


def normalize_party_codes(df: pd.DataFrame, party_columns: list = None) -> pd.DataFrame:
    """
    Normalize party codes in specified columns to standard single-letter uppercase format.

    This ensures consistency for downstream analyses by standardizing various forms of party
    names/codes to the canonical single-letter codes used throughout the project:
    - 'D' for Democratic
    - 'R' for Republican
    - Other standard codes for minor parties

    Args:
        df: DataFrame containing party code column(s)
        party_columns: List of column names to normalize. If None, defaults to
                       ['party_of_appointing_president'] if it exists

    Returns:
        DataFrame with normalized party codes
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # If no columns specified, use default if it exists
    if party_columns is None:
        party_columns = []
        if "party_of_appointing_president" in result_df.columns:
            party_columns.append("party_of_appointing_president")

    # Dictionary for normalizing various forms of party names/codes
    party_mapping = {
        # Democratic variants
        "democrat": "D",
        "democratic": "D",
        "democracy": "D",
        "democrat party": "D",
        "democratic party": "D",
        "d": "D",
        "dem": "D",
        # Republican variants
        "republican": "R",
        "republicans": "R",
        "republican party": "R",
        "r": "R",
        "rep": "R",
        # Other parties
        "federalist": "F",
        "f": "F",
        "whig": "W",
        "w": "W",
        "union": "U",
        "u": "U",
        "democratic-republican": "DR",
        "dr": "DR",
        "independent": "I",
        "i": "I",
        "libertarian": "L",
        "l": "L",
        "green": "G",
        "g": "G",
    }

    # Process each column
    normalized_count = 0
    for col in party_columns:
        if col not in result_df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue

        # Only process non-NA values
        mask = result_df[col].notna()
        if not mask.any():
            logger.info(f"No values to normalize in column '{col}'")
            continue

        # Count values before normalization
        before_count = result_df[col].value_counts().to_dict()

        # Track unmapped values and their row indices
        unmapped_values = {}

        # Apply normalization function to each value
        def normalize(idx_val_tuple):
            idx, val = idx_val_tuple
            if pd.isna(val):
                return val

            # Convert to string, strip whitespace, lowercase for consistent lookup
            val_str = str(val).strip().lower()

            # Check if value is in mapping dictionary
            mapped_val = party_mapping.get(val_str, None)

            # If not in dictionary, record it and return original value
            if mapped_val is None:
                if val not in unmapped_values:
                    unmapped_values[val] = []
                unmapped_values[val].append(idx)
                return val

            # Return mapped value
            return mapped_val

        # Apply normalization with row indices
        result_df.loc[mask, col] = [
            normalize((idx, val))
            for idx, val in zip(result_df.loc[mask].index, result_df.loc[mask, col])
        ]

        # Log unmapped values with their row indices
        if unmapped_values:
            for val, indices in unmapped_values.items():
                sample_indices = indices[:5]  # Limit to first 5 indices to avoid excessive logging
                more_text = f" and {len(indices) - 5} more rows" if len(indices) > 5 else ""
                logger.warning(
                    f"Unmapped party value '{val}' found in column '{col}' at rows {sample_indices}{more_text} - kept as-is"
                )

        # Count normalized values
        after_count = result_df[col].value_counts().to_dict()
        normalized_count += sum(mask)

        # Log the changes
        logger.info(f"Normalized {sum(mask)} party codes in column '{col}'")
        logger.debug(f"Before normalization: {before_count}")
        logger.debug(f"After normalization: {after_count}")

    return result_df


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
    while first_monday.weekday() != 0:  # 0 = Monday
        first_monday += timedelta(days=1)
    # Tuesday following that
    return first_monday + timedelta(days=1)


def _convert_to_date(d):
    """Convert datetime or pandas Timestamp to date if needed."""
    if hasattr(d, "date") and callable(getattr(d, "date")):
        return d.date()
    return d


def _presidency_record(d) -> tuple[int, date, str, str]:
    """
    Return (president_number, first_inauguration_date_of_that_presidency, party, full_name)
    for the date supplied.  Raises ValueError if out of range.

    Args:
        d: date, datetime or pandas Timestamp object

    Returns:
        Tuple of (president_number, first_inauguration_date, party, full_name)
    """
    # Convert datetime/Timestamp to date if needed
    d_date = _convert_to_date(d)

    for number, start, end, party, full_name in PRESIDENCY_DATES:
        if d_date >= start and (end is None or d_date < end):
            return number, start, party, full_name
    raise ValueError(f"Date {d} out of presidency table range.")


# ---------------------------------------------------------------------------
# 3.  FUNCTIONS TO HELP A JUPYTER NOTEBOOK
# ---------------------------------------------------------------------------


def president_number(d) -> int:
    """38 for Gerald Ford, 46 for Joe Biden, …

    Args:
        d: date, datetime or pandas Timestamp object
    """
    return _presidency_record(d)[0]


def president_party(d) -> str:
    """
    Returns party affiliation of the president in office on the given date.
    'D' for Democratic, 'R' for Republican, etc.

    Args:
        d: date, datetime or pandas Timestamp object

    Returns:
        String representing party ('D', 'R', etc.)
    """
    return _presidency_record(d)[2]


def president_name(d) -> str:
    """
    Returns the full name of the president in office on the given date.
    Format is "First Middle Last" to match the appointing_president column.

    Args:
        d: date, datetime or pandas Timestamp object

    Returns:
        String containing the president's full name in First Middle Last format
    """
    return _presidency_record(d)[3]


def presidential_term_index(d) -> int:
    """
    1 for the first four-year term of that president,
    2 for the second, 3/4 for FDR, etc.

    Args:
        d: date, datetime or pandas Timestamp object
    """
    d_date = _convert_to_date(d)
    number, first_inauguration, _, _ = _presidency_record(d_date)  # Unpack ignoring party and name
    # Years (minus a day fudge) elapsed since first inauguration
    years_elapsed = (d_date - first_inauguration).days // 365.2425
    return int(years_elapsed // 4) + 1


def days_into_current_term(d) -> int:
    """1-based: inauguration day itself returns 1.

    Args:
        d: date, datetime or pandas Timestamp object
    """
    d_date = _convert_to_date(d)
    number, first_inauguration, _, _ = _presidency_record(d_date)  # Unpack ignoring party and name
    term_idx = presidential_term_index(d_date)
    term_start = first_inauguration.replace(year=first_inauguration.year + 4 * (term_idx - 1))
    return (d_date - term_start).days + 1


def days_until_next_presidential_election(d) -> int:
    """
    Count of days from `d` (exclusive) up to the **next** presidential-year
    Election Day (years divisible by 4). Returns 0 if `d` *is* Election Day.

    Args:
        d: date, datetime or pandas Timestamp object
    """
    d_date = _convert_to_date(d)
    year = d_date.year
    while year % 4 != 0 or d_date > _election_day(year):
        year += 1
    return (_election_day(year) - d_date).days


def days_until_next_midterm_election(d) -> int:
    """
    Days until the next even-year Election Day **that is not a presidential year**.

    Args:
        d: date, datetime or pandas Timestamp object
    """
    d_date = _convert_to_date(d)
    year = d_date.year
    while year % 2 != 0 or year % 4 == 0 or d_date > _election_day(year):
        year += 1
    return (_election_day(year) - d_date).days


# ----------------  CONGRESS & SESSION  -------------------------------------

CONGRESS_FIRST_YEAR = 1789  # First Congress began Mar-4-1789
CONGRESS_START_MONTH_PRE20TH = 3  # March 4 before 1935
CONGRESS_START_MONTH_20TH = 1  # Jan   3 starting 1935


def congress_number(d) -> int:
    """
    118  → Jan-3-2023 to Jan-3-2025, etc.
    Formula: each Congress spans two calendar years, starting in odd years.

    Args:
        d: date, datetime or pandas Timestamp object
    """
    d_date = _convert_to_date(d)
    if d_date < date(1789, 3, 4):
        raise ValueError("Congress did not yet exist.")
    start_year = d_date.year if d_date.year % 2 else d_date.year - 1
    return (start_year - CONGRESS_FIRST_YEAR) // 2 + 1


def congress_session(d) -> int:
    """
    1  → first (odd-year) session
    2  → second (even-year) session
    3  → anything else (special/emergency)

    Args:
        d: date, datetime or pandas Timestamp object
    """
    d_date = _convert_to_date(d)
    cong_start_year = d_date.year if d_date.year % 2 else d_date.year - 1
    if d_date.year == cong_start_year:
        return 1
    elif d_date.year == cong_start_year + 1:
        return 2
    else:
        return 3
