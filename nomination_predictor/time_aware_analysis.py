import ast
from datetime import date, timedelta
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

from loguru import logger
import pandas as pd
from tqdm import tqdm

from nomination_predictor.config import INTERIM_DATA_DIR, RAW_DATA_DIR

# ---------------------------------------------------------------------------
# 1.  PRESIDENTIAL TERMS  (simple flat list; None = still in office)
#    tuple = (president_number, first_inauguration_date, last_day_in_office_or_None)
# ---------------------------------------------------------------------------

PRESIDENT_TERMS = [
    # Modern era (you can extend backward easily)
    (46, date(2021, 1, 20), None),  # Joseph R. Biden
    (45, date(2017, 1, 20), date(2021, 1, 20)),
    (44, date(2009, 1, 20), date(2017, 1, 20)),
    (43, date(2001, 1, 20), date(2009, 1, 20)),
    (42, date(1993, 1, 20), date(2001, 1, 20)),
    (41, date(1989, 1, 20), date(1993, 1, 20)),
    (40, date(1981, 1, 20), date(1989, 1, 20)),
    (39, date(1977, 1, 20), date(1981, 1, 20)),
    (38, date(1974, 8, 9), date(1977, 1, 20)),
    (37, date(1969, 1, 20), date(1974, 8, 9)),
    (36, date(1963, 11, 22), date(1969, 1, 20)),
    (35, date(1961, 1, 20), date(1963, 11, 22)),
    (34, date(1953, 1, 20), date(1961, 1, 20)),
    (33, date(1945, 4, 12), date(1953, 1, 20)),
    (32, date(1933, 3, 4), date(1945, 4, 12)),
    (31, date(1929, 3, 4), date(1933, 3, 4)),
    (30, date(1923, 8, 2), date(1929, 3, 4)),
    (29, date(1921, 3, 4), date(1923, 8, 2)),
    (28, date(1913, 3, 4), date(1921, 3, 4)),
    (27, date(1909, 3, 4), date(1913, 3, 4)),
    (26, date(1901, 9, 14), date(1909, 3, 4)),
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
    while first_monday.weekday() != 0:  # 0 = Monday
        first_monday += timedelta(days=1)
    # Tuesday following that
    return first_monday + timedelta(days=1)


@lru_cache(None)
def _president_record(d: date) -> tuple[int, date]:
    """
    Return (president_number, first_inauguration_date_of_that_presidency)
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
    number, first_inauguration = _president_record(d)
    # Years (minus a day fudge) elapsed since first inauguration
    years_elapsed = (d - first_inauguration).days // 365.2425
    return int(years_elapsed // 4) + 1


def days_into_current_term(d: date) -> int:
    """1-based: inauguration day itself returns 1."""
    number, first_inauguration = _president_record(d)
    term_idx = presidential_term_index(d)
    term_start = first_inauguration.replace(year=first_inauguration.year + 4 * (term_idx - 1))
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

CONGRESS_FIRST_YEAR = 1789  # First Congress began Mar-4-1789
CONGRESS_START_MONTH_PRE20TH = 3  # March 4 before 1935
CONGRESS_START_MONTH_20TH = 1  # Jan   3 starting 1935


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


def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = INTERIM_DATA_DIR / "features.csv",
    # -----------------------------------------
) -> None:
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------



def snapshot_education(education_df: pd.DataFrame, nid: str, cutoff_date: date) -> Dict[str, str]:
    """
    Return highest degree information for a judge prior to cutoff date.

    Args:
        education_df: Education dataframe with nid, sequence, degree, school, degree_year
        nid: Judge's NID to filter by
        cutoff_date: Only consider education completed by this date

    Returns:
        Dictionary with highest degree and school information
    """
    # Filter by NID
    edu = education_df[education_df["nid"] == nid].copy()

    if edu.empty:
        return {"highest_degree": "", "highest_degree_school": "", "highest_degree_year": None}

    # Convert degree_year to integer and ensure it's before cutoff_date
    edu["degree_year"] = pd.to_numeric(edu["degree_year"], errors="coerce")
    edu = edu.dropna(subset=["degree_year"])
    edu = edu[edu["degree_year"] <= cutoff_date.year]

    if edu.empty:
        return {"highest_degree": "", "highest_degree_school": "", "highest_degree_year": None}

    # Map degree types to numerical levels for sorting
    degree_levels = {
        "J.D.": 5,  # Juris Doctor
        "LL.M.": 5,  # Master of Laws
        "LL.B.": 4,  # Bachelor of Laws
        "Ph.D.": 6,  # Doctorate
        "S.J.D.": 6,  # Doctor of Juridical Science
        "M.D.": 5,  # Medical Doctor
        "M.A.": 3,  # Master of Arts
        "M.S.": 3,  # Master of Science
        "M.B.A.": 3,  # Master of Business Administration
        "B.A.": 2,  # Bachelor of Arts
        "B.S.": 2,  # Bachelor of Science
        "A.A.": 1,  # Associate of Arts
        "A.S.": 1,  # Associate of Science
    }

    # Apply degree level mapping (default to 0 if not found)
    edu["degree_level"] = edu["degree"].apply(
        lambda d: next(
            (level for deg, level in degree_levels.items() if deg.lower() in str(d).lower()), 0
        )
    )

    # Sort by degree level (highest first), then by year (most recent first), then by sequence
    edu = edu.sort_values(
        ["degree_level", "degree_year", "sequence"], ascending=[False, False, False]
    )

    # Get highest degree
    top = edu.iloc[0]

    return {
        "highest_degree": str(top["degree"]),
        "highest_degree_school": str(top["school"]),
        "highest_degree_year": int(top["degree_year"])
        if not pd.isna(top["degree_year"])
        else None,
        "has_law_degree": any(
            "j.d." in str(d).lower() or "ll.b." in str(d).lower() for d in edu["degree"]
        ),
    }


def extract_years_from_career(career_text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract start and end years from career text description.

    Args:
        career_text: Text description of career position with years

    Returns:
        Tuple of (start_year, end_year) - end_year may be None for current positions
    """
    if pd.isna(career_text):
        return None, None

    # Pattern for years at end of text (e.g., "1990-1995" or "1990-present" or "2000-")
    year_pattern = r"(\d{4})(?:-(\d{4}|present|\s*))$"
    match = re.search(year_pattern, career_text)

    if match:
        start_year = int(match.group(1))
        end_year_str = match.group(2) if match.group(2) else None

        if end_year_str and end_year_str.isdigit():
            end_year = int(end_year_str)
        else:
            # Handle "present" or missing end year as current year
            end_year = None
        return start_year, end_year

    # Alternative pattern when years are comma-separated (e.g., "Law clerk, Judge Smith, 1990-1991")
    alt_pattern = r"(\d{4})-(\d{4}|present)"
    match = re.search(alt_pattern, career_text)

    if match:
        start_year = int(match.group(1))
        end_year_str = match.group(2)
        end_year = int(end_year_str) if end_year_str.isdigit() else None
        return start_year, end_year

    return None, None


def snapshot_career(career_df: pd.DataFrame, nid: str, cutoff_date: date) -> Dict[str, Any]:
    """
    Compute career snapshot features for a judge prior to cutoff date.

    Args:
        career_df: Career dataframe with nid, sequence, professional_career columns
        nid: Judge's NID to filter by
        cutoff_date: Only consider career positions up to this date

    Returns:
        Dictionary with aggregated career metrics
    """
    # Filter by NID
    careers = career_df[career_df["nid"] == nid].copy()

    if careers.empty:
        return {
            "years_private_practice": 0,
            "has_govt_experience": False,
            "has_prosecutor_experience": False,
            "has_public_defender_experience": False,
            "has_military_experience": False,
            "has_law_professor_experience": False,
            "has_state_judge_experience": False,
            "has_federal_clerk_experience": False,
        }

    # Extract years from career descriptions
    careers["start_year"], careers["end_year"] = zip(
        *careers["professional_career"].apply(extract_years_from_career)
    )

    # Filter by cutoff date
    careers = careers[careers["start_year"].notna()]
    careers = careers[careers["start_year"] <= cutoff_date.year]

    if careers.empty:
        return {
            "years_private_practice": 0,
            "has_govt_experience": False,
            "has_prosecutor_experience": False,
            "has_public_defender_experience": False,
            "has_military_experience": False,
            "has_law_professor_experience": False,
            "has_state_judge_experience": False,
            "has_federal_clerk_experience": False,
        }

    # Calculate years in different roles
    cutoff_year = cutoff_date.year

    # Define patterns for different career types
    career_patterns = {
        "private_practice": r"private practice|law firm|partner|associate",
        "govt_experience": r"department of justice|attorney general|solicitor general|government|federal|state",
        "prosecutor": r"prosecutor|district attorney|u\.s\. attorney|state attorney",
        "public_defender": r"public defender|legal aid|legal services",
        "military": r"army|navy|marine|air force|military|jag|judge advocate",
        "law_professor": r"professor|faculty|lecturer|instructor|taught|law school",
        "state_judge": r"judge|justice|court|judicial|magistrate|commissioner",
        "federal_clerk": r"clerk|clerked|clerkship",
    }

    # Calculate years in private practice
    private_practice_years = 0
    for _, row in careers.iterrows():
        career_text = str(row["professional_career"]).lower()
        start_year = row["start_year"]

        # Skip if missing start year
        if pd.isna(start_year):
            continue

        # Use actual end year or cutoff year, whichever is earlier
        end_year = row["end_year"]
        if pd.isna(end_year) or end_year > cutoff_year:
            end_year = cutoff_year

        if re.search(career_patterns["private_practice"], career_text):
            years = end_year - start_year
            if years > 0:
                private_practice_years += years

    # Check for experience in different areas
    career_text_combined = " ".join(careers["professional_career"].fillna("").str.lower())

    return {
        "years_private_practice": private_practice_years,
        "has_govt_experience": bool(
            re.search(career_patterns["govt_experience"], career_text_combined)
        ),
        "has_prosecutor_experience": bool(
            re.search(career_patterns["prosecutor"], career_text_combined)
        ),
        "has_public_defender_experience": bool(
            re.search(career_patterns["public_defender"], career_text_combined)
        ),
        "has_military_experience": bool(
            re.search(career_patterns["military"], career_text_combined)
        ),
        "has_law_professor_experience": bool(
            re.search(career_patterns["law_professor"], career_text_combined)
        ),
        "has_state_judge_experience": bool(
            re.search(career_patterns["state_judge"], career_text_combined)
        ),
        "has_federal_clerk_experience": bool(
            re.search(career_patterns["federal_clerk"], career_text_combined)
        ),
    }




def is_nominee_confirmed(
    nominee_row: pd.Series, nominations_df: pd.DataFrame, citation_key: str = "citation"
) -> bool:
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


def filter_confirmed_nominees(
    nominees_df: pd.DataFrame, nominations_df: pd.DataFrame, citation_key: str = "citation"
) -> pd.DataFrame:
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
        lambda row: is_nominee_confirmed(row, nominations_df, citation_key), axis=1
    )

    # Filter the DataFrame
    confirmed_nominees = nominees_df[is_confirmed].copy()
    logger.info(
        f"Filtered {len(nominees_df)} nominees to {len(confirmed_nominees)} confirmed nominees"
    )

    return confirmed_nominees