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
# 1.  PRESIDENTIAL TERMS WITH PARTY AFFILIATION AND FULL NAMES
#    tuple = (president_number, first_inauguration_date, last_day_in_office, party, full_name)
#    party: 'D' = Democratic, 'R' = Republican, 'DR' = Democratic-Republican, 
#           'F' = Federalist, 'W' = Whig, 'U' = Union, etc.
#    full_name: First Middle Last format, for filling appointing_president_(1) column
# ---------------------------------------------------------------------------

PRESIDENCY_DATES = [
    # Format: (number, start_date, end_date, party, full_name)
    # Names formatted as First Middle Last to match appointing_president_(1) column format
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
    (20, date(1881, 3, 4,), date(1881, 9, 19), "R", "James Abram Garfield"),
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
    """Fill missing values in the appointing_president_(1) column using nomination date."""
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if appointing_president_(1) column is already fully populated
    if not result_df["appointing_president_(1)"].isna().any():
        logger.info("No missing appointing president values detected - dataframe already fully populated")
        return result_df
    
    # Only process rows where appointing_president_(1) is missing but nomination_date is available
    mask = result_df["appointing_president_(1)"].isna() & result_df["receiveddate"].notna()
    
    # Check if we have any rows to process
    if not mask.any():
        logger.info("No rows available to fill: either no missing appointing president values or missing reception dates")
        return result_df
    
    # Apply the president_name function to get the appointing president
    result_df.loc[mask, "appointing_president_(1)"] = \
        result_df.loc[mask, "receiveddate"].apply(president_name)
    
    # Log the number of filled values
    num_filled = mask.sum()
    logger.info(f"Filled {num_filled} missing appointing president values using nomination dates")
    
    return result_df


def fill_missing_party_of_appointing_presidents(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the party_of_appointing_president_(1) column using nomination date."""
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if party_of_appointing_president_(1) column exists, create if not
    if "party_of_appointing_president_(1)" not in result_df.columns:
        result_df["party_of_appointing_president_(1)"] = None
        logger.info("Created party_of_appointing_president_(1) column which was missing")
    
    # Check if party_of_appointing_president_(1) column is already fully populated
    if not result_df["party_of_appointing_president_(1)"].isna().any():
        logger.info("No missing party of appointing president values detected - dataframe already fully populated")
        return result_df
    
    # Only process rows where party_of_appointing_president_(1) is missing but receiveddate is available
    mask = result_df["party_of_appointing_president_(1)"].isna() & result_df["receiveddate"].notna()
    
    # Check if we have any rows to process
    if not mask.any():
        logger.info("No rows available to fill: either no missing party values or missing reception dates")
        return result_df
    
    # Apply the president_party function to get the party of the appointing president
    result_df.loc[mask, "party_of_appointing_president_(1)"] = \
        result_df.loc[mask, "receiveddate"].apply(president_party)
    
    # Log the number of filled values
    num_filled = mask.sum()
    logger.info(f"Filled {num_filled} missing party of appointing president values using nomination dates")
    
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
                       ['party_of_appointing_president_(1)'] if it exists
    
    Returns:
        DataFrame with normalized party codes
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If no columns specified, use default if it exists
    if party_columns is None:
        party_columns = []
        if "party_of_appointing_president_(1)" in result_df.columns:
            party_columns.append("party_of_appointing_president_(1)")
    
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
        "g": "G"
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
    if hasattr(d, 'date') and callable(getattr(d, 'date')):
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
    Format is "First Middle Last" to match the appointing_president_(1) column.
    
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
