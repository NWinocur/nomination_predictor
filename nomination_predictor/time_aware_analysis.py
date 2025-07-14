from datetime import date, timedelta
from pathlib import Path
import re

from loguru import logger
import pandas as pd

from nomination_predictor.config import INTERIM_DATA_DIR, RAW_DATA_DIR


def merge_congress_fjc(
    cong_noms_df: pd.DataFrame,
    fjc_judges_df: pd.DataFrame,
    fjc_demographics_df: pd.DataFrame,
    fjc_education_df: pd.DataFrame,
    fjc_career_df: pd.DataFrame,
    fjc_service_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Congress nominations with FJC judge data (education, career, demographics, service),
    including only education and job history records up to each nomination's received date.
    Flattens multiple records into columns and adds career experience flags.
    """
    # Ensure the nomination date is in datetime format for comparisons
    if "receiveddate" in cong_noms_df.columns:
        cong_noms_df["receiveddate"] = pd.to_datetime(cong_noms_df["receiveddate"])
    elif "received_date" in cong_noms_df.columns:
        # If a different column name is used for the received date
        cong_noms_df["received_date"] = pd.to_datetime(cong_noms_df["received_date"])
        cong_noms_df["receiveddate"] = cong_noms_df["received_date"]
    else:
        raise KeyError("Nominations DataFrame must have a 'receiveddate' column for cutoff dates.")

    # Copy and index the base nominations dataframe to keep one row per nomination
    base_df = cong_noms_df.copy().reset_index(drop=False)
    base_df.rename(
        columns={"index": "nomination_index"}, inplace=True
    )  # use index as unique nomination ID

    # ----------------- Education Records (Time-Restricted) -----------------
    edu_df = fjc_education_df.copy()
    # Convert degree_year to datetime (year precision) and extract year as integer
    edu_df["degree_year"] = pd.to_datetime(edu_df["degree_year"], errors="coerce")
    edu_df["degree_year_year"] = edu_df["degree_year"].dt.year

    # Join nominations with all education records by nid
    edu_merge = pd.merge(
        base_df[["nomination_index", "nid", "receiveddate"]], edu_df, on="nid", how="left"
    )
    # Restrict to education entries completed on or before the nomination's year
    edu_merge = edu_merge[
        edu_merge["degree_year_year"].notna()
        & (edu_merge["degree_year_year"] <= edu_merge["receiveddate"].dt.year)
    ]
    # Note: Using year-only comparison – if a degree was earned in the same year as the nomination,
    # we assume it was completed by nomination time. This may include degrees finished later in the year,
    # but exact dates are not available (fallback assumption).

    # Sort and flatten all qualifying education records per nomination
    edu_merge.sort_values(["nomination_index", "sequence"], inplace=True)
    edu_merge["edu_rank"] = (
        edu_merge.groupby("nomination_index").cumcount() + 1
    )  # rank degrees in chronological order

    # Pivot degree info into separate columns (degree_1, school_1, degree_year_1, etc.)
    degree_wide = edu_merge.pivot(index="nomination_index", columns="edu_rank", values="degree")
    school_wide = edu_merge.pivot(index="nomination_index", columns="edu_rank", values="school")
    degyear_wide = edu_merge.pivot(
        index="nomination_index", columns="edu_rank", values="degree_year_year"
    )
    # Rename pivoted columns with sequential suffixes
    degree_wide.columns = [f"degree_{int(col)}" for col in degree_wide.columns]
    school_wide.columns = [f"school_{int(col)}" for col in school_wide.columns]
    degyear_wide.columns = [f"degree_year_{int(col)}" for col in degyear_wide.columns]
    # Combine all education columns
    edu_wide = pd.concat([degree_wide, school_wide, degyear_wide], axis=1).reset_index()

    # ----------------- Professional Career Records (Time-Restricted) -----------------
    career_df = fjc_career_df.copy()

    # Extract start and end year from each career description using regex
    def extract_years(text: str):
        if pd.isna(text):
            return (None, None)
        text_str = str(text).strip().lower()
        # Pattern for a year range at the end of the string (e.g. "2010-2015", "2010-present", or "2010-")
        match = re.search(r"(\d{4})(?:-(\d{4}|present|\s*))$", text_str)
        if match:
            start_year = int(match.group(1))
            end_str = match.group(2)
            end_year = int(end_str) if end_str and end_str.isdigit() else None
            return (start_year, end_year)
        # Alternative pattern for a year range elsewhere in the text (e.g. "1990-1995" within the string)
        match2 = re.search(r"(\d{4})-(\d{4}|present)", text_str)
        if match2:
            start_year = int(match2.group(1))
            end_str = match2.group(2)
            end_year = int(end_str) if end_str.isdigit() else None
            return (start_year, end_year)
        # If no year found, return (None, None)
        return (None, None)

    # Apply extraction to every career entry
    year_pairs = career_df["professional_career"].map(extract_years).tolist()
    start_years, end_years = zip(*year_pairs) if len(year_pairs) > 0 else ([], [])
    career_df["start_year"] = start_years
    career_df["end_year"] = end_years

    # Join nominations with all career records by nid
    career_merge = pd.merge(
        base_df[["nomination_index", "nid", "receiveddate"]], career_df, on="nid", how="left"
    )
    # Restrict to career entries that started on or before the nomination year
    career_merge = career_merge[
        career_merge["start_year"].notna()
        & (career_merge["start_year"] <= career_merge["receiveddate"].dt.year)
    ]
    # (If a position began in the same year as nomination, we include it as ongoing by that year.)

    # Sort and flatten all qualifying career records per nomination
    career_merge.sort_values(["nomination_index", "sequence"], inplace=True)
    career_merge["job_rank"] = (
        career_merge.groupby("nomination_index").cumcount() + 1
    )  # rank jobs chronologically

    # Pivot career descriptions into separate columns (professional_career_1, _2, etc.)
    career_wide = career_merge.pivot(
        index="nomination_index", columns="job_rank", values="professional_career"
    )
    career_wide.columns = [f"professional_career_{int(col)}" for col in career_wide.columns]
    career_wide = career_wide.reset_index()

    # ----------------- Career Experience Flags & Metrics -----------------
    flags_list = []
    for nom_idx, group in career_merge.groupby("nomination_index"):
        # Combine all career text for this nomination
        combined_text = " ".join(group["professional_career"].fillna("").astype(str)).lower()
        # Define regex patterns for experience categories
        patterns = {
            "private_practice": r"private practice|law firm|partner|associate",
            "govt_experience": r"department of justice|attorney general|solicitor general|government|federal|state",
            "prosecutor": r"prosecutor|district attorney|u\.s\. attorney|state attorney",
            "public_defender": r"public defender|legal aid|legal services",
            "military": r"\barmy\b|\bnavy\b|\bmarine\b|\bair force\b|military|jag|judge advocate",
            "law_professor": r"professor|faculty|lecturer|instructor|taught|law school",
            "state_judge": r"\bjudge\b|\bjustice\b|court|judicial|magistrate|commissioner",
            "federal_clerk": r"clerk\b|clerked|clerkship",
        }
        cutoff_year = int(group["receiveddate"].dt.year.iloc[0])
        # Calculate total years in private practice up to the cutoff date
        total_private_years = 0
        for _, row in group.iterrows():
            role_text = str(row["professional_career"]).lower()
            sy = row["start_year"]
            ey = row["end_year"]
            if pd.isna(sy):
                continue
            # If no end year or role extends beyond cutoff, assume it continues through the nomination year
            if pd.isna(ey) or ey > cutoff_year:
                ey = cutoff_year
            if re.search(patterns["private_practice"], role_text):
                # Add the span of years in private practice (inclusive of start year, exclusive of end year)
                try:
                    years = int(ey) - int(sy)
                except Exception:
                    years = 0
                if years > 0:
                    total_private_years += years
        # Compile flags for each category based on combined text of careers
        flags_list.append(
            {
                "nomination_index": nom_idx,
                "years_private_practice": total_private_years,
                "has_govt_experience": bool(re.search(patterns["govt_experience"], combined_text)),
                "has_prosecutor_experience": bool(
                    re.search(patterns["prosecutor"], combined_text)
                ),
                "has_public_defender_experience": bool(
                    re.search(patterns["public_defender"], combined_text)
                ),
                "has_military_experience": bool(re.search(patterns["military"], combined_text)),
                "has_law_professor_experience": bool(
                    re.search(patterns["law_professor"], combined_text)
                ),
                "has_state_judge_experience": bool(
                    re.search(patterns["state_judge"], combined_text)
                ),
                "has_federal_clerk_experience": bool(
                    re.search(patterns["federal_clerk"], combined_text)
                ),
            }
        )
    flags_df = pd.DataFrame(flags_list)

    # ----------------- Federal Judicial Service (Time-Restricted) -----------------
    service_df = fjc_service_df.copy()
    # Ensure commission_date is datetime for filtering
    if "commission_date" in service_df.columns:
        service_df["commission_date"] = pd.to_datetime(
            service_df["commission_date"], errors="coerce"
        )
    # Join nominations with all federal judicial service records by nid
    service_merge = pd.merge(
        base_df[["nomination_index", "nid", "receiveddate"]], service_df, on="nid", how="left"
    )
    # Include only service records where the commission date was on or before the nomination date
    service_merge = service_merge[
        service_merge["commission_date"].notna()
        & (service_merge["commission_date"] <= service_merge["receiveddate"])
    ]
    # This captures any prior federal judgeships the nominee held by that time.
    # (Service records for judgeships acquired *after* the nomination are excluded to prevent future data leakage.)

    service_merge.sort_values(["nomination_index", "sequence"], inplace=True)
    service_merge["service_rank"] = service_merge.groupby("nomination_index").cumcount() + 1

    # Pivot key service fields for each prior appointment into columns (e.g., court_name_1, court_name_2, ...)
    service_fields = [
        "court_name",
        "court_type",
        "appointment_title",
        "appointing_president",
        "party_of_appointing_president",
        "nomination_date",
        "confirmation_date",
        "commission_date",
        "termination_date",
    ]
    service_wide = pd.DataFrame()
    if not service_merge.empty:
        service_wide = service_merge.pivot(
            index="nomination_index", columns="service_rank", values=service_fields
        )
        # Flatten multi-index columns (field, rank) into single level names
        service_wide.columns = [f"{field}_{int(rank)}" for field, rank in service_wide.columns]
        service_wide = service_wide.reset_index()

    # ----------------- Merging All Components -----------------
    # Start with base nominations and incrementally merge each set of features
    merged = base_df.merge(edu_wide, on="nomination_index", how="left")
    merged = merged.merge(career_wide, on="nomination_index", how="left")
    merged = merged.merge(flags_df, on="nomination_index", how="left")
    merged = merged.merge(service_wide, on="nomination_index", how="left")

    # Fill missing experience fields for nominations with no prior career records
    merged["years_private_practice"] = merged["years_private_practice"].fillna(0)
    for col in [
        "has_govt_experience",
        "has_prosecutor_experience",
        "has_public_defender_experience",
        "has_military_experience",
        "has_law_professor_experience",
        "has_state_judge_experience",
        "has_federal_clerk_experience",
    ]:
        merged[col] = merged[col].fillna(False)

    # Merge FJC core judge info (demographics and judge data)
    # Drop redundant name and birth/death fields from demographics to avoid duplication
    demo_df = fjc_demographics_df.drop(
        columns=[
            "last_name",
            "first_name",
            "middle_name",
            "suffix",
            "birth_month",
            "birth_day",
            "birth_year",
            "birth_city",
            "birth_state",
            "death_month",
            "death_day",
            "death_year",
            "death_city",
            "death_state",
        ],
        errors="ignore",
    )
    merged = merged.merge(fjc_judges_df, on="nid", how="left", suffixes=("", "_fjcjudges"))
    merged = merged.merge(demo_df, on="nid", how="left", suffixes=("", "_fjcdemo"))
    # (All available demographic fields like gender and race are now included.
    # Duplicate personal info fields were removed as they were explicitly redundant between sources.)

    # Drop the temporary index used for pivoting
    merged.drop(columns=["nomination_index"], inplace=True)
    return merged


def main(
    input_path: str = RAW_DATA_DIR / "dataset.csv",
    output_path: str = INTERIM_DATA_DIR / "merged_time_aware.csv",
) -> None:
    """Read source datasets, perform time-aware merge, and save the result."""
    logger.info("Reading input datasets...")
    # Read Congress nominations data (with NID and receiveddate) and FJC datasets
    cong_noms_df = pd.read_csv(input_path)
    fjc_judges_df = pd.read_csv(RAW_DATA_DIR / "fjc_judges.csv")
    fjc_demo_df = pd.read_csv(RAW_DATA_DIR / "fjc_demographics.csv")
    fjc_edu_df = pd.read_csv(RAW_DATA_DIR / "fjc_education.csv")
    fjc_career_df = pd.read_csv(RAW_DATA_DIR / "fjc_professional_career.csv")
    fjc_service_df = pd.read_csv(RAW_DATA_DIR / "fjc_federal_judicial_service.csv")

    logger.info("Performing time-aware merge of Congress and FJC data...")
    merged_df = merge_congress_fjc(
        cong_noms_df, fjc_judges_df, fjc_demo_df, fjc_edu_df, fjc_career_df, fjc_service_df
    )

    # Save the merged dataframe to the interim data directory
    output_path = Path(output_path)  # ensure Path object
    merged_df.to_csv(output_path, index=False)
    logger.success(f"Time-aware merged dataset saved to {output_path} (shape: {merged_df.shape})")


# ---------------------------------------------------------------------------
# 1.  PRESIDENTIAL TERMS  (simple flat list; None = still in office)
#    tuple = (president_number, first_inauguration_date, last_day_in_office_or_None)
# ---------------------------------------------------------------------------

PRESIDENCY_DATES = [
    (47, date(2025, 1, 20), date(2029, 1, 20)), # using projected end of single term is the simplest way I can think to program for
    (46, date(2021, 1, 20), date(2025, 1, 20)),
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
    (25, date(1897, 3, 4), date(1901, 9, 14)),
    (24, date(1893, 3, 4), date(1897, 3, 4)),
    (23, date(1889, 3, 4), date(1893, 3, 4)),
    (22, date(1885, 3, 4), date(1889, 3, 4)),
    (21, date(1881, 9, 19), date(1885, 3, 4)),
    (20, date(1881, 3, 4,), date(1881, 9, 19)),
    (19, date(1877, 3, 4), date(1881, 3, 4)),
    (18, date(1869, 3, 4), date(1877, 3, 4)),
    (17, date(1865, 4, 15), date(1869, 3, 4)),
    (16, date(1861, 3, 4), date(1865, 4, 15)),
    (15, date(1857, 3, 4), date(1861, 3, 4)),
    (14, date(1853, 3, 4), date(1857, 3, 4)),
    # if you're able to retrieve judicial & congressional data via API dating back further, congratulations on your data gathering!
    # Until then this is good enough for the data I've been able to obtain. 
]

# Pre-sort newest-first so we can break quickly in lookups
PRESIDENCY_DATES.sort(key=lambda t: t[1], reverse=True)


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
    if hasattr(d, 'date') and callable(getattr(d, 'date')):
        return d.date()
    return d


def _presidency_record(d) -> tuple[int, date]:
    """
    Return (president_number, first_inauguration_date_of_that_presidency)
    for the date supplied.  Raises ValueError if out of range.
    
    Args:
        d: date, datetime or pandas Timestamp object
        
    Returns:
        Tuple of (president_number, first_inauguration_date)
    """
    # Convert datetime/Timestamp to date if needed
    d_date = _convert_to_date(d)
    
    for number, start, end in PRESIDENCY_DATES:
        if d_date >= start and (end is None or d_date < end):
            return number, start
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


def presidential_term_index(d) -> int:
    """
    1 for the first four-year term of that president,
    2 for the second, 3/4 for FDR, etc.
    
    Args:
        d: date, datetime or pandas Timestamp object
    """
    d_date = _convert_to_date(d)
    number, first_inauguration = _presidency_record(d_date)
    # Years (minus a day fudge) elapsed since first inauguration
    years_elapsed = (d_date - first_inauguration).days // 365.2425
    return int(years_elapsed // 4) + 1


def days_into_current_term(d) -> int:
    """1-based: inauguration day itself returns 1.
    
    Args:
        d: date, datetime or pandas Timestamp object
    """
    d_date = _convert_to_date(d)
    number, first_inauguration = _presidency_record(d_date)
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
