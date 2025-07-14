from datetime import date
from pathlib import Path
import re

from loguru import logger
import pandas as pd

from nomination_predictor.config import INTERIM_DATA_DIR, RAW_DATA_DIR


def merge_congress_fjc(cong_noms_df: pd.DataFrame,
                       fjc_judges_df: pd.DataFrame,
                       fjc_demographics_df: pd.DataFrame,
                       fjc_education_df: pd.DataFrame,
                       fjc_career_df: pd.DataFrame,
                       fjc_service_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Congress nominations with FJC judge data (education, career, demographics, service),
    including only education and job history records up to each nomination's received date.
    Flattens multiple records into columns and adds career experience flags.
    """
    # Ensure the nomination date is in datetime format for comparisons
    if 'receiveddate' in cong_noms_df.columns:
        cong_noms_df['receiveddate'] = pd.to_datetime(cong_noms_df['receiveddate'])
    elif 'received_date' in cong_noms_df.columns:
        # If a different column name is used for the received date
        cong_noms_df['received_date'] = pd.to_datetime(cong_noms_df['received_date'])
        cong_noms_df['receiveddate'] = cong_noms_df['received_date']
    else:
        raise KeyError("Nominations DataFrame must have a 'receiveddate' column for cutoff dates.")
    
    # Copy and index the base nominations dataframe to keep one row per nomination
    base_df = cong_noms_df.copy().reset_index(drop=False)
    base_df.rename(columns={'index': 'nomination_index'}, inplace=True)  # use index as unique nomination ID
    
    # ----------------- Education Records (Time-Restricted) -----------------
    edu_df = fjc_education_df.copy()
    # Convert degree_year to datetime (year precision) and extract year as integer
    edu_df['degree_year'] = pd.to_datetime(edu_df['degree_year'], errors='coerce')
    edu_df['degree_year_year'] = edu_df['degree_year'].dt.year
    
    # Join nominations with all education records by nid
    edu_merge = pd.merge(base_df[['nomination_index', 'nid', 'receiveddate']],
                         edu_df, on='nid', how='left')
    # Restrict to education entries completed on or before the nomination's year
    edu_merge = edu_merge[edu_merge['degree_year_year'].notna() &
                          (edu_merge['degree_year_year'] <= edu_merge['receiveddate'].dt.year)]
    # Note: Using year-only comparison – if a degree was earned in the same year as the nomination,
    # we assume it was completed by nomination time. This may include degrees finished later in the year,
    # but exact dates are not available (fallback assumption).
    
    # Sort and flatten all qualifying education records per nomination
    edu_merge.sort_values(['nomination_index', 'sequence'], inplace=True)
    edu_merge['edu_rank'] = edu_merge.groupby('nomination_index').cumcount() + 1  # rank degrees in chronological order
    
    # Pivot degree info into separate columns (degree_1, school_1, degree_year_1, etc.)
    degree_wide = edu_merge.pivot(index='nomination_index', columns='edu_rank', values='degree')
    school_wide = edu_merge.pivot(index='nomination_index', columns='edu_rank', values='school')
    degyear_wide = edu_merge.pivot(index='nomination_index', columns='edu_rank', values='degree_year_year')
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
        match = re.search(r'(\d{4})(?:-(\d{4}|present|\s*))$', text_str)
        if match:
            start_year = int(match.group(1))
            end_str = match.group(2)
            end_year = int(end_str) if end_str and end_str.isdigit() else None
            return (start_year, end_year)
        # Alternative pattern for a year range elsewhere in the text (e.g. "1990-1995" within the string)
        match2 = re.search(r'(\d{4})-(\d{4}|present)', text_str)
        if match2:
            start_year = int(match2.group(1))
            end_str = match2.group(2)
            end_year = int(end_str) if end_str.isdigit() else None
            return (start_year, end_year)
        # If no year found, return (None, None)
        return (None, None)
    
    # Apply extraction to every career entry
    year_pairs = career_df['professional_career'].map(extract_years).tolist()
    start_years, end_years = zip(*year_pairs) if len(year_pairs) > 0 else ([], [])
    career_df['start_year'] = start_years
    career_df['end_year'] = end_years
    
    # Join nominations with all career records by nid
    career_merge = pd.merge(base_df[['nomination_index', 'nid', 'receiveddate']],
                             career_df, on='nid', how='left')
    # Restrict to career entries that started on or before the nomination year
    career_merge = career_merge[career_merge['start_year'].notna() &
                                (career_merge['start_year'] <= career_merge['receiveddate'].dt.year)]
    # (If a position began in the same year as nomination, we include it as ongoing by that year.)
    
    # Sort and flatten all qualifying career records per nomination
    career_merge.sort_values(['nomination_index', 'sequence'], inplace=True)
    career_merge['job_rank'] = career_merge.groupby('nomination_index').cumcount() + 1  # rank jobs chronologically
    
    # Pivot career descriptions into separate columns (professional_career_1, _2, etc.)
    career_wide = career_merge.pivot(index='nomination_index', columns='job_rank', values='professional_career')
    career_wide.columns = [f"professional_career_{int(col)}" for col in career_wide.columns]
    career_wide = career_wide.reset_index()
    
    # ----------------- Career Experience Flags & Metrics -----------------
    flags_list = []
    for nom_idx, group in career_merge.groupby('nomination_index'):
        # Combine all career text for this nomination
        combined_text = " ".join(group['professional_career'].fillna("").astype(str)).lower()
        # Define regex patterns for experience categories
        patterns = {
            "private_practice": r"private practice|law firm|partner|associate",
            "govt_experience": r"department of justice|attorney general|solicitor general|government|federal|state",
            "prosecutor": r"prosecutor|district attorney|u\.s\. attorney|state attorney",
            "public_defender": r"public defender|legal aid|legal services",
            "military": r"\barmy\b|\bnavy\b|\bmarine\b|\bair force\b|military|jag|judge advocate",
            "law_professor": r"professor|faculty|lecturer|instructor|taught|law school",
            "state_judge": r"\bjudge\b|\bjustice\b|court|judicial|magistrate|commissioner",
            "federal_clerk": r"clerk\b|clerked|clerkship"
        }
        cutoff_year = int(group['receiveddate'].dt.year.iloc[0])
        # Calculate total years in private practice up to the cutoff date
        total_private_years = 0
        for _, row in group.iterrows():
            role_text = str(row['professional_career']).lower()
            sy = row['start_year']; ey = row['end_year']
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
        flags_list.append({
            "nomination_index": nom_idx,
            "years_private_practice": total_private_years,
            "has_govt_experience": bool(re.search(patterns["govt_experience"], combined_text)),
            "has_prosecutor_experience": bool(re.search(patterns["prosecutor"], combined_text)),
            "has_public_defender_experience": bool(re.search(patterns["public_defender"], combined_text)),
            "has_military_experience": bool(re.search(patterns["military"], combined_text)),
            "has_law_professor_experience": bool(re.search(patterns["law_professor"], combined_text)),
            "has_state_judge_experience": bool(re.search(patterns["state_judge"], combined_text)),
            "has_federal_clerk_experience": bool(re.search(patterns["federal_clerk"], combined_text))
        })
    flags_df = pd.DataFrame(flags_list)
    
    # ----------------- Federal Judicial Service (Time-Restricted) -----------------
    service_df = fjc_service_df.copy()
    # Ensure commission_date is datetime for filtering
    if 'commission_date' in service_df.columns:
        service_df['commission_date'] = pd.to_datetime(service_df['commission_date'], errors='coerce')
    # Join nominations with all federal judicial service records by nid
    service_merge = pd.merge(base_df[['nomination_index', 'nid', 'receiveddate']],
                              service_df, on='nid', how='left')
    # Include only service records where the commission date was on or before the nomination date
    service_merge = service_merge[service_merge['commission_date'].notna() &
                                  (service_merge['commission_date'] <= service_merge['receiveddate'])]
    # This captures any prior federal judgeships the nominee held by that time. 
    # (Service records for judgeships acquired *after* the nomination are excluded to prevent future data leakage.)
    
    service_merge.sort_values(['nomination_index', 'sequence'], inplace=True)
    service_merge['service_rank'] = service_merge.groupby('nomination_index').cumcount() + 1
    
    # Pivot key service fields for each prior appointment into columns (e.g., court_name_1, court_name_2, ...)
    service_fields = ['court_name', 'court_type', 'appointment_title',
                      'appointing_president', 'party_of_appointing_president',
                      'nomination_date', 'confirmation_date', 'commission_date', 'termination_date']
    service_wide = pd.DataFrame()
    if not service_merge.empty:
        service_wide = service_merge.pivot(index='nomination_index', columns='service_rank', values=service_fields)
        # Flatten multi-index columns (field, rank) into single level names
        service_wide.columns = [f"{field}_{int(rank)}" for field, rank in service_wide.columns]
        service_wide = service_wide.reset_index()
    
    # ----------------- Merging All Components -----------------
    # Start with base nominations and incrementally merge each set of features
    merged = base_df.merge(edu_wide, on='nomination_index', how='left')
    merged = merged.merge(career_wide, on='nomination_index', how='left')
    merged = merged.merge(flags_df, on='nomination_index', how='left')
    merged = merged.merge(service_wide, on='nomination_index', how='left')
    
    # Fill missing experience fields for nominations with no prior career records
    merged['years_private_practice'] = merged['years_private_practice'].fillna(0)
    for col in ["has_govt_experience", "has_prosecutor_experience", "has_public_defender_experience",
                "has_military_experience", "has_law_professor_experience",
                "has_state_judge_experience", "has_federal_clerk_experience"]:
        merged[col] = merged[col].fillna(False)
    
    # Merge FJC core judge info (demographics and judge data)
    # Drop redundant name and birth/death fields from demographics to avoid duplication
    demo_df = fjc_demographics_df.drop(columns=[
        'last_name', 'first_name', 'middle_name', 'suffix',
        'birth_month', 'birth_day', 'birth_year', 'birth_city', 'birth_state',
        'death_month', 'death_day', 'death_year', 'death_city', 'death_state'
    ], errors='ignore')
    merged = merged.merge(fjc_judges_df, on='nid', how='left', suffixes=('', '_fjcjudges'))
    merged = merged.merge(demo_df, on='nid', how='left', suffixes=('', '_fjcdemo'))
    # (All available demographic fields like gender and race are now included. 
    # Duplicate personal info fields were removed as they were explicitly redundant between sources.)
    
    # Drop the temporary index used for pivoting
    merged.drop(columns=['nomination_index'], inplace=True)
    return merged

def main(input_path: str = RAW_DATA_DIR / "dataset.csv",
         output_path: str = INTERIM_DATA_DIR / "merged_time_aware.csv") -> None:
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
    merged_df = merge_congress_fjc(cong_noms_df, fjc_judges_df, fjc_demo_df,
                                   fjc_edu_df, fjc_career_df, fjc_service_df)
    
    # Save the merged dataframe to the interim data directory
    output_path = Path(output_path)  # ensure Path object
    merged_df.to_csv(output_path, index=False)
    logger.success(f"Time-aware merged dataset saved to {output_path} (shape: {merged_df.shape})")
