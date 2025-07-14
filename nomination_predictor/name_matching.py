from loguru import logger
from nameparser import HumanName
import pandas as pd


def split_names(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """
    Add three lowercase columns: first, last, middle_initial.

    Args:
        df: DataFrame containing names to parse
        name_col: Column name containing full names

    Returns:
        DataFrame with added first, last, and mi (middle initial) columns
    """
    parsed = df[name_col].fillna("").apply(HumanName)
    df["first"] = parsed.apply(lambda n: n.first.lower().strip())
    df["last"] = parsed.apply(lambda n: n.last.lower().strip())
    df["mi"] = parsed.apply(lambda n: (n.middle[:1] or "").lower())
    return df


def perform_exact_name_matching(
    congress_df: pd.DataFrame,
    fjc_df: pd.DataFrame,
    congress_name_col: str = "full_name_from_description",
    fjc_name_col: str = "judge_name",
    nid_col: str = "nid",
) -> pd.DataFrame:
    """
    Performs exact string matching between Congress.gov and FJC names using a
    vectorized pandas merge approach.

    Matching priority: last name -> first name -> middle initial.

    Args:
        congress_df: DataFrame with Congress.gov nomination data
        fjc_df: DataFrame with FJC judge data
        congress_name_col: Column containing Congress nominee names
        fjc_name_col: Column containing FJC judge names
        nid_col: Column containing unique FJC identifier

    Returns:
        DataFrame with Congress records matched to FJC NIDs and match information
    """
    logger.info(
        f"Starting exact name matching with {len(congress_df)} Congress records and {len(fjc_df)} FJC records"
    )

    # Step 1: Add first, last, and middle initial columns to both dataframes
    cong = split_names(congress_df.copy(), congress_name_col)
    fjc = split_names(fjc_df.copy(), fjc_name_col)

    # Ensure we have a unique identifier for congress records
    if "congress_index" not in cong.columns:
        cong["congress_index"] = cong.index

    # Only keep one row per NID in the FJC dataframe to avoid duplicates
    # from different service records for the same person
    fjc_unique = fjc[[nid_col, "first", "last", "mi", fjc_name_col]].drop_duplicates(
        subset=[nid_col]
    )

    # Step 2: First-pass join on last name and first name
    logger.info("Performing first-pass join on last and first name")
    m1 = cong.merge(
        fjc_unique, on=["last", "first"], how="left", suffixes=("", "_fjc"), indicator=True
    )

    # Step 3: Check for ambiguity - multiple FJC matches for a single Congress record
    ambiguity_mask = m1.duplicated(subset=["congress_index"], keep=False) & (m1["_merge"] == "both")

    # Records that matched exactly one FJC record on last+first name
    is_unambiguous = m1.loc[~ambiguity_mask & (m1["_merge"] == "both")].copy()
    is_unambiguous["match_type"] = "first_and_last_name"
    is_unambiguous["ambiguous"] = False

    # Records that matched multiple FJC records - need further disambiguation
    pending_ambiguity_determination = m1.loc[ambiguity_mask].copy()

    # Step 4: Second-pass join to try to disambiguate with middle initial
    logger.info(
        f"Found {len(pending_ambiguity_determination) // 2} ambiguous matches, attempting middle initial disambiguation"
    )

    # Use a full join with middle initial to capture all permutations
    resolved = pending_ambiguity_determination.merge(
        fjc_unique, on=["last", "first", "mi"], how="inner", suffixes=("", "_r")
    ).drop_duplicates(subset=["congress_index"])

    # Create a mask for records that got disambiguated with middle initial
    disambiguated_indices = set(resolved["congress_index"])

    # Extract congress records that were disambiguated with middle initial
    middle_initial_resolved = resolved.copy()
    middle_initial_resolved["match_type"] = "first_middle_last_name"
    middle_initial_resolved["ambiguous"] = False

    # Identify congress records that are still ambiguous
    still_ambiguous_indices = set(pending_ambiguity_determination["congress_index"]) - disambiguated_indices
    still_ambiguous = pending_ambiguity_determination[pending_ambiguity_determination["congress_index"].isin(still_ambiguous_indices)].copy()

    # For ambiguous matches, we need to ensure we only have one row per congress_index/nid pair
    still_ambiguous = still_ambiguous.drop_duplicates(subset=["congress_index", nid_col])
    still_ambiguous["match_type"] = "last_name_only"
    still_ambiguous["ambiguous"] = True

    # Step 5: Create the final result dataframe with all matches
    logger.info("Creating final results dataframe")

    # Select and rename columns to match expected output format
    result_columns = {
        "congress_index": "congress_index",
        congress_name_col: "congress_name",
        fjc_name_col: "fjc_name",
        nid_col: "nid",
        "match_type": "match_type",
        "ambiguous": "ambiguous",
    }

    # Combine unambiguous and ambiguous results
    results = []

    # Add unambiguous first+last name matches
    if len(is_unambiguous) > 0:
        results.append(is_unambiguous[list(result_columns.keys())])

    # Add middle initial resolved matches
    if len(middle_initial_resolved) > 0:
        results.append(middle_initial_resolved[list(result_columns.keys())])

    # Add still ambiguous matches
    if len(still_ambiguous) > 0:
        results.append(still_ambiguous[list(result_columns.keys())])

    # Combine all results
    if results:
        final_results = pd.concat(results, ignore_index=True)
    else:
        # Create empty DataFrame with proper columns if no matches were found
        final_results = pd.DataFrame(columns=list(result_columns.values()))

    # Rename columns to expected output format
    final_results = final_results.rename(columns=result_columns)

    # Log summary statistics
    total_matches = len(final_results)
    unambiguous_count = len(final_results[~final_results["ambiguous"]])
    ambiguous_count = len(final_results[final_results["ambiguous"]])

    logger.info(f"Name matching complete: {total_matches} total matches")
    logger.info(f"  - {unambiguous_count} unambiguous matches")
    logger.info(f"  - {ambiguous_count} ambiguous matches")

    return final_results
