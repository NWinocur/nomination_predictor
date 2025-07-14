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

    # DIAGNOSTIC: Check if we found any matches at all
    found_matches = m1[m1["_merge"] == "both"]
    logger.info(f"Found {len(found_matches)} total records with last+first name matches")

    if found_matches.empty:
        # Try last-name-only matches to diagnose data issues
        logger.info(
            "NO last+first name matches found. Checking last-name-only matches for diagnosis..."
        )
        last_only = cong.merge(fjc_unique, on=["last"], how="inner", suffixes=("", "_fjc"))
        logger.info(f"Found {len(last_only)} last-name-only matches")
        if not last_only.empty:
            logger.info("Showing up to first 10 last-name-only matches:")
            sample_cols = ["congress_index", congress_name_col, fjc_name_col, nid_col]
            logger.info(last_only[sample_cols].head(10))

    # Step 3: Check for ambiguity - multiple FJC matches for a single Congress record
    ambiguity_mask = m1.duplicated(subset=["congress_index"], keep=False) & (
        m1["_merge"] == "both"
    )

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
    if not pending_ambiguity_determination.empty:
        logger.info("Samples of pending ambiguous rows")
        logger.info(pending_ambiguity_determination.sample(5))

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
    still_ambiguous_indices = (
        set(pending_ambiguity_determination["congress_index"]) - disambiguated_indices
    )
    still_ambiguous = pending_ambiguity_determination[
        pending_ambiguity_determination["congress_index"].isin(still_ambiguous_indices)
    ].copy()

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
