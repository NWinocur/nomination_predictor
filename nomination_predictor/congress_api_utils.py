import re

import pandas as pd

from nomination_predictor.features import clean_name, create_full_name_from_parts


def extract_court_from_description(description: str) -> str:
    """
    Extract court information from nomination description.

    Format typically follows: "Full name, of the US_state, to be a/an position_title
    for/of the court_name, (for a/an optional term limit), vice predecessor_name, reason."

    Args:
        description: The nomination description text

    Returns:
        Extracted court name or empty string if not found
    """
    if pd.isna(description):
        return ""

    # Common patterns for court names in descriptions
    court_patterns = [
        r"to be (?:a |an )?(?:United States |U\.S\. )?(?:District |Circuit |Chief )?Judge for the (.*?(?:District|Circuit).*?)(?:,|\.|\s+for a| vice)",
        r"to be (?:a |an )?(?:United States |U\.S\. )?(?:District |Circuit |Chief )?Judge of the (.*?(?:District|Circuit).*?)(?:,|\.|\s+for a| vice)",
        r"to be (?:a |an )?(?:Judge|Justice) of the (.*?Court.*?)(?:,|\.|\s+for a| vice)",
        r"to be (?:a |an )?(?:Associate|Chief) (?:Judge|Justice) of the (.*?Court.*?)(?:,|\.|\s+for a| vice)",
    ]

    # Try each pattern
    for pattern in court_patterns:
        match = re.search(pattern, description)
        if match:
            return match.group(1).strip()

    # Fallback - look for any court reference
    match = re.search(
        r"(?:District|Circuit|Court|Tribunal) (?:of|for) (?:the )?([A-Za-z\s]+)", description
    )
    if match:
        return match.group(0).strip()

    return ""


def enrich_congress_nominees_dataframe(
    nominees_df: pd.DataFrame, nominations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds derived name fields to the Congress nominees DataFrame.

    Args:
        nominees_df: Congress nominees DataFrame with firstname, lastname, middlename columns
        nominations_df: Nominations DataFrame to join for additional fields

    Returns:
        DataFrame with additional name fields
    """
    # Create a copy to avoid modifying the original
    enriched_df = nominees_df.copy()

    # Create full name from components
    enriched_df["full_name"] = enriched_df.apply(
        lambda row: create_full_name_from_parts(
            row.get("firstname"), row.get("middlename"), row.get("lastname"), row.get("suffix")
        ),
        axis=1,
    )

    # Add cleaned full name
    enriched_df["full_name_clean"] = enriched_df["full_name"].apply(clean_name)

    # For backwards compatibility, ensure first/middle/last exist
    # Note: We'll use the values from the API if they exist
    if "first" not in enriched_df.columns:
        enriched_df["first"] = enriched_df["firstname"].fillna("")

    if "middle" not in enriched_df.columns:
        enriched_df["middle"] = enriched_df["middlename"].fillna("")

    if "last" not in enriched_df.columns:
        enriched_df["last"] = enriched_df["lastname"].fillna("")

    # If nominations dataframe is provided, merge relevant fields
    if nominations_df is not None and "citation" in enriched_df.columns:
        # Create a mapping from citation to relevant nomination fields
        nom_info = {}
        for _, row in nominations_df.iterrows():
            if "citation" in row and "description" in row:
                citation = row["citation"]
                # Extract organization if available
                org = row.get("nominee_organization", "")
                if pd.isna(org) or not org:
                    org = row.get("organization", "")

                # Extract court from description
                court_from_desc = extract_court_from_description(row["description"])

                # Get received date (nomination date)
                nomination_date = row.get("receiveddate", "")

                nom_info[citation] = {
                    "organization": org,
                    "court_from_description": court_from_desc,
                    "description": row["description"],
                    "nomination_date": nomination_date,
                }

        # Add extracted fields
        def get_nomination_info(citation, field):
            if citation in nom_info:
                return nom_info[citation].get(field, "")
            return ""

        enriched_df["organization"] = enriched_df["citation"].apply(
            lambda c: get_nomination_info(c, "organization")
        )

        enriched_df["court_from_description"] = enriched_df["citation"].apply(
            lambda c: get_nomination_info(c, "court_from_description")
        )

        enriched_df["nomination_description"] = enriched_df["citation"].apply(
            lambda c: get_nomination_info(c, "description")
        )

        # Add nomination date
        enriched_df["nomination_date"] = enriched_df["citation"].apply(
            lambda c: get_nomination_info(c, "nomination_date")
        )

        # Add normalized court field - try to use court from description first, fall back to organization
        enriched_df["court_clean"] = enriched_df.apply(
            lambda row: normalized_court(row["court_from_description"])
            if row.get("court_from_description")
            else normalized_court(row.get("organization", "")),
            axis=1,
        )
    elif "organization" in enriched_df.columns:
        # If organization is directly in nominees_df (unlikely based on user's clarification)
        enriched_df["court_clean"] = enriched_df["organization"].apply(normalized_court)

    return enriched_df


def normalized_court(text: str) -> str:
    """
    Normalizes court names for consistency.

    Args:
        text: Court name string

    Returns:
        Normalized court name
    """
    if pd.isna(text):
        return ""
    text = text.upper().replace("UNITED STATES", "").replace("U.S.", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text
