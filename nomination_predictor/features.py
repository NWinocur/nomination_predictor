"""This file contains code to create features for modeling, i.e. data which wasn't in the raw data but which is being made available for the model.
It shall never output to "raw" data.  It is meant to take raw or interim data as input and deliver interim data as output.


Examples of data feature-generation include utility functions for feature-engineering U.S. federal political timelines, or capabilities such as:

Given a vacancy is on record as having existed in a specified court, with a named incumbent judge, a known reason the vacancy occurred, and a known date the vacancy began, when a vacancy with equivalent data is found on other months, then it shall be treated as the same vacancy incident.
Given earlier records for a vacancy incident lack a nominee and nomination date, when a later record for the same vacancy incident has a nominee and nomination date, then the nominee and nomination date shall be merged onto that vacancy incident's records in interim data.
Given a vacancy is on record as having existed in a specified court, with a named incumbent judge, when a vacancy with equivalent court location and incumbent is found to have occurred with a different reason and/or vacancy date (e.g. a nomination is withdrawn and reopened), then the two vacancy incidents shall be treated as each deserving of their own unique record in interim data.
Given a nomination is on record as having occurred on a specified date, when filling in date-dependent feature data (e.g. which President performed the nomination), then the interim data shall be updated to include date-inferred data (e.g. a numeric ordinal indicator identifying that President who was in office on the date of nomination.
"""

import ast
from datetime import date, timedelta
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

from loguru import logger
import pandas as pd
from thefuzz import fuzz
from tqdm import tqdm

from nomination_predictor.config import INTERIM_DATA_DIR, RAW_DATA_DIR


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names in a DataFrame: convert to lowercase and replace spaces with underscores.

    Args:
        df: DataFrame to normalize

    Returns:
        DataFrame with normalized column names
    """
    if df is None or df.empty:
        return df

    # Create a mapping of old names to new names
    column_mapping = {col: col.casefold().replace(" ", "_") for col in df.columns}

    # Rename columns using the mapping
    return df.rename(columns=column_mapping)


def normalize_all_dataframes(dataframes_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Normalize column names for all DataFrames in a dictionary.

    Args:
        dataframes_dict: Dictionary of DataFrames with string keys

    Returns:
        Dictionary with same keys but DataFrames having normalized column names
    """
    normalized_dfs: Dict[str, pd.DataFrame] = {}

    for name, df in dataframes_dict.items():
        if df is not None and not df.empty:
            normalized_dfs[name] = normalize_dataframe_columns(df)
        else:
            normalized_dfs[name] = df

    return normalized_dfs



def extract_court_and_position(nominations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract court and position information from nomination descriptions.

    Args:
        nominations_df: Dataframe containing nomination data with 'description' field

    Returns:
        DataFrame with additional extracted fields
    """
    result_df = nominations_df.copy()

    # Extract court and position information from descriptions
    courts = []
    positions = []
    states = []

    position_pattern = r"to be (?:a |an )?(.*?)(?: for| of| to)(?: the)? (.*?)(?:,|\.|\s+vice)"
    state_pattern = r"of (.*?)(?:,|\.|\s+to be)"

    for desc in result_df["description"]:
        if pd.isna(desc):
            courts.append("")
            positions.append("")
            states.append("")
            continue

        # Try to extract position and court
        pos_match = re.search(position_pattern, desc)
        if pos_match:
            position = pos_match.group(1).strip()
            court = pos_match.group(2).strip()
            positions.append(position)
            courts.append(court)
        else:
            positions.append("")
            courts.append("")

        # Try to extract state
        state_match = re.search(state_pattern, desc)
        if state_match:
            state = state_match.group(1).strip()
            states.append(state)
        else:
            states.append("")

    # Add extracted fields to dataframe
    result_df["extracted_position"] = positions
    result_df["extracted_court"] = courts
    result_df["extracted_state"] = states

    return result_df




def merge_nominees_with_nominations(
    nominees_df: pd.DataFrame, nominations_df: pd.DataFrame, join_key: str = "citation"
) -> pd.DataFrame:
    """
    Merge nominees with their corresponding nominations.

    Args:
        nominees_df: Dataframe with nominee information
        nominations_df: Dataframe with nomination information
        join_key: Key to use for joining the dataframes

    Returns:
        Merged dataframe
    """
    if join_key not in nominees_df.columns or join_key not in nominations_df.columns:
        raise ValueError(f"Join key '{join_key}' not found in both dataframes")

    # Extract court and position information
    enhanced_nominations = extract_court_and_position(nominations_df)

    # Merge dataframes
    merged = nominees_df.merge(
        enhanced_nominations, on=join_key, how="left", suffixes=("_nominee", "_nomination")
    )

    return merged


def filter_non_judicial_nominations(
    frame_with_position_titles: pd.DataFrame, non_judicial_titles: list[str] = None
) -> pd.DataFrame:
    """
    Filter out non-judicial nominations based on position titles.

    Args:
        frame_with_position_titles: DataFrame containing nomination records with 'nominees_0_positiontitle' and 'citation' columns
        non_judicial_titles: List of strings indicating non-judicial position titles

    Returns:
        Tuple of (filtered_nominations_df, filtered_nominees_df)
    """
    if non_judicial_titles is None:
        non_judicial_titles = [
            "Attorney",
            "Board",
            "Commission",
            "Director",
            "Marshal",
            "Assistant",
            "Representative",
            "Secretary of",
            "Member of",
        ]

    # Make copies to avoid SettingWithCopyWarning
    df_copy = frame_with_position_titles.copy()

    # Find citations of rows with non-judicial titles
    non_judicial_mask = df_copy["nominees_0_positiontitle"].str.contains(
        "|".join(non_judicial_titles), na=False
    )
    citations_to_drop = df_copy.loc[non_judicial_mask, "citation"].unique()

    # Log the number of non-judicial nominations being removed
    logger.info(f"Found {len(citations_to_drop)} unique citations with non-judicial titles")

    # Filter out the non-judicial nominations
    filtered_nominations = df_copy[~df_copy["citation"].isin(citations_to_drop)]

    # Log the results
    logger.info(
        f"Removed {len(df_copy) - len(filtered_nominations)}/{len(df_copy)} corresponding records"
    )

    return filtered_nominations


def _safe_parse_json(json_str: str) -> dict:
    """
    Robustly parse a JSON string that might not be strictly JSON compliant (e.g., with single quotes).

    Args:
        json_str: String representation of JSON/dict, possibly with single quotes

    Returns:
        Parsed dictionary
    """
    if not isinstance(json_str, str):
        # If it's already a dictionary, return it directly
        return json_str if isinstance(json_str, dict) else {}

    try:
        # First try standard JSON parsing
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # If that fails, try ast.literal_eval which can handle Python dict literals
            return ast.literal_eval(json_str)
        except (SyntaxError, ValueError) as e:
            # For debugging: Show the problematic string
            if len(json_str) > 200:
                logger.warning(
                    f"Could not parse JSON-like string (showing first 200 chars): {json_str[:200]}..., error: {e}"
                )
            else:
                logger.warning(f"Could not parse JSON-like string: {json_str}, error: {e}")

            # Last resort: Try to normalize common patterns
            try:
                # Replace single quotes with double quotes
                s = json_str.replace("'", '"')
                # Fix True/False/None literals
                s = s.replace("True", "true").replace("False", "false").replace("None", "null")
                return json.loads(s)
            except Exception as e:
                # If all attempts fail, log the error and return empty dict
                logger.error(f"Could not parse JSON string: {json_str[:100]}...")
                return {}


def flatten_json_dataframe(
    df: pd.DataFrame, json_col: str, max_list_index: int = 10, separator: str = "_"
) -> pd.DataFrame:
    """
    Flatten a DataFrame containing JSON/dictionary data in a specified column into a wide format with
    a single row per record, converting nested structures into column names.

    This function handles both dictionary and list structures in the JSON data:
    - Dictionary keys become column prefixes
    - List items are enumerated up to max_list_index
    - Nested structures are recursively flattened

    Args:
        df: DataFrame containing JSON data
        json_col: Name of the column containing JSON data (either as string or parsed dict/list)
        max_list_index: Maximum number of list items to extract as separate columns (default: 10)
        separator: String to use as separator between nested keys (default: '_')

    Returns:
        DataFrame with flattened columns, where all nested JSON structures are expanded
        into separate columns with appropriate prefixes.

    Example:
        Input JSON: {"nominees": [{"firstName": "John", "lastName": "Smith"}], "congress": 117}
        Output columns:
            - congress
            - nominees_0_firstName
            - nominees_0_lastName
    """
    if json_col not in df.columns:
        raise ValueError(f"Column '{json_col}' not found in DataFrame")

    # Create a new DataFrame to hold the flattened results
    result_df = pd.DataFrame(index=df.index)

    # Copy over any non-JSON columns from the original DataFrame
    for col in df.columns:
        if col != json_col:
            result_df[col] = df[col]

    # Process each row
    logger.info(f"Flattening JSON data from column '{json_col}' in {len(df)} rows")

    for idx, row in df.iterrows():
        try:
            # Get the JSON data and parse it if needed
            json_data = row[json_col]
            if isinstance(json_data, str):
                json_data = _safe_parse_json(json_data)

            # Flatten this specific JSON structure
            flat_dict = {}
            _flatten_json_recursive(
                json_data, flat_dict, prefix="", max_list_index=max_list_index, separator=separator
            )

            # Add flattened columns to result DataFrame
            for key, value in flat_dict.items():
                if key not in result_df.columns:
                    result_df[key] = None  # Create the column if it doesn't exist
                result_df.at[idx, key] = value

        except Exception as e:
            logger.error(f"Error flattening JSON in row {idx}: {e}")
            # Keep the original JSON in case of error
            result_df.at[idx, json_col] = row[json_col]

    logger.info(
        f"Flattening complete. Original columns: {len(df.columns)}, New columns: {len(result_df.columns)}"
    )
    return result_df


def _flatten_json_recursive(
    obj: Any, flat_dict: Dict[str, Any], prefix: str, max_list_index: int, separator: str
) -> None:
    """
    Recursively flatten a nested JSON object into a flat dictionary.

    Args:
        obj: The nested object to flatten (can be dict, list, or scalar)
        flat_dict: The output dictionary to store flattened key-value pairs
        prefix: Current prefix for keys (from parent objects)
        max_list_index: Maximum number of list items to extract
        separator: String to use between nested keys
    """
    # Handle different types of objects
    if isinstance(obj, dict):
        # Process dictionary: each key becomes a column prefix
        for key, value in obj.items():
            # Create new key with prefix
            new_key = f"{prefix}{key}" if prefix == "" else f"{prefix}{separator}{key}"

            if isinstance(value, (dict, list)):
                # Recursively flatten nested structures
                _flatten_json_recursive(value, flat_dict, new_key, max_list_index, separator)
            else:
                # Store leaf values directly
                flat_dict[new_key] = value

    elif isinstance(obj, list):
        # Process list: enumerate items up to max_list_index
        for i, item in enumerate(obj):
            if i >= max_list_index:
                # Stop if we've reached the maximum list items to extract
                break

            # Create new key with index
            new_key = f"{prefix}{separator}{i}" if prefix else f"{i}"

            if isinstance(item, (dict, list)):
                # Recursively flatten nested structures
                _flatten_json_recursive(item, flat_dict, new_key, max_list_index, separator)
            else:
                # Store leaf values directly
                flat_dict[new_key] = item

    else:
        # Handle scalar values directly
        flat_dict[prefix] = obj




def extract_last_name(full_name: str) -> str:
    """
    Extract last name from a full name, handling suffixes and special cases.
    
    Args:
        full_name: Full name string
        
    Returns:
        Last name or empty string if extraction fails
    """
    if not full_name or pd.isna(full_name):
        return ""
    
    # Remove suffixes like Jr., Sr., III, etc.
    name = str(full_name).strip()
    suffixes = [' jr', ' sr', ' ii', ' iii', ' iv', ' v', ' esq']
    
    for suffix in suffixes:
        if suffix in name.lower():
            # Find the suffix position and cut off the name there
            pos = name.lower().find(suffix)
            name = name[:pos]
    
    # Handle "Last, First Middle" format
    if ',' in name:
        return name.split(',')[0].strip()
    
    # For "First Middle Last" format, take the last word
    return name.split()[-1].strip()




def clean_name(name: str) -> str:
    """
    Clean and normalize a name string.

    Args:
        name: Name string to clean

    Returns:
        Cleaned name string
    """
    if pd.isna(name):
        return ""
    name = str(name).casefold()
    name = re.sub(r"[\.,]", "", name)  # drop punctuation
    name = re.sub(r"\s+", " ", name).strip()
    return name


def split_name(name: str) -> Tuple[str, str, str]:
    """
    Very naive splitter: returns first, middle (maybe empty), last

    Args:
        name: Full name to split

    Returns:
        Tuple of (first_name, middle_name, last_name)
    """
    parts = clean_name(name).split()
    if not parts:
        return "", "", ""
    if len(parts) == 1:
        return parts[0], "", ""
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return parts[0], " ".join(parts[1:-1]), parts[-1]


def create_full_name_from_parts(
    first_name: Optional[str] = None,
    middle_name: Optional[str] = None,
    last_name: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """
    Creates a full name from individual name parts.

    Args:
        first_name: First name
        middle_name: Middle name
        last_name: Last name
        suffix: Name suffix (e.g., Jr., Sr., III)

    Returns:
        Combined full name string
    """
    components = []

    if first_name and not pd.isna(first_name):
        components.append(str(first_name).strip())

    if middle_name and not pd.isna(middle_name):
        components.append(str(middle_name).strip())

    if last_name and not pd.isna(last_name):
        components.append(str(last_name).strip())

    full_name = " ".join(components)

    if suffix and not pd.isna(suffix):
        suffix_str = str(suffix).strip()
        if suffix_str:
            full_name = f"{full_name} {suffix_str}"

    return full_name


def extract_court_from_description(description: str) -> str:
    """
    DEPRECATED: this function was trying to do too much all at once, and has been replaced by multiple smaller functions.
    
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



def merge_nominees_onto_nominations(
    nominations_df: pd.DataFrame, nominees_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge nominees dataframe onto nominations dataframe using URL as the join key.

    The process works in two steps:
    1. Extract URL from nominees' "request" column (which contains a dict)
    2. Merge nominees onto nominations using the URL field

    Args:
        nominations_df: DataFrame containing nomination data with 'nominees_0_url' column
        nominees_df: DataFrame containing nominee data with 'request' column

    Returns:
        DataFrame with nominees data merged onto nominations
    """
    # Step 1: Extract URLs from nominees dataframe's "request" column
    nominees_with_url = nominees_df.copy()

    # Create nominees_0_url column from request dictionary
    try:
        # Extract URL from request string or dictionary
        nominees_with_url["nominees_0_url"] = nominees_with_url["request"].apply(
            lambda req_str: _safe_parse_json(req_str).get("url")
            if isinstance(req_str, str)
            else req_str.get("url")
            if isinstance(req_str, dict)
            else None
        )

        # Check if URLs were successfully extracted
        url_count = nominees_with_url["nominees_0_url"].notna().sum()
        logger.info(
            f"Extracted {url_count} URLs from nominees request column ({url_count / len(nominees_with_url):.1%} of rows)"
        )

        if url_count == 0:
            logger.warning("No URLs extracted from nominees. Check the 'request' column format.")
            return nominations_df  # Return original dataframe if no URLs found

    except KeyError:
        logger.error("'request' column not found in nominees dataframe")
        return nominations_df
    except Exception as e:
        logger.error(f"Error extracting URLs from nominees: {e}")
        return nominations_df

    # Step 2: Merge nominees onto nominations using URL as the join key
    try:
        # Verify nominations_df has the URL column
        if "nominees_0_url" not in nominations_df.columns:
            logger.error("'nominees_0_url' column not found in nominations dataframe")
            return nominations_df

        # Count non-null values before merge
        nom_url_count = nominations_df["nominees_0_url"].notna().sum()
        logger.info(
            f"Nominations dataframe has {nom_url_count} non-null URLs ({nom_url_count / len(nominations_df):.1%} of rows)"
        )

        # Perform left merge
        merged_df = nominations_df.merge(
            nominees_with_url.drop(
                columns=["request"]
            ),  # Remove the request column as we already extracted the URL
            on="nominees_0_url",
            how="left",
            suffixes=("", "_nominee"),
        )

        # Report merge results
        match_count = merged_df["nominees_0_url"].notna().sum()
        logger.info(f"Merged dataframe has {len(merged_df)} rows")
        logger.info(
            f"Successfully matched {match_count} nominations with nominees ({match_count / len(merged_df):.1%})"
        )

        return merged_df

    except Exception as e:
        logger.error(f"Error during merge: {e}")
        return nominations_df


def convert_judge_name_format_from_last_to_first_to_first_then_last(judge_name):
    """
    Convert judge name from "lastname, firstname middlename (suffix)" format
    to "firstname lastname middlename suffix" format.

    Args:
        judge_name: String in format "lastname, firstname middlename (suffix)"

    Returns:
        String in format "firstname lastname middlename suffix"
    """
    if not isinstance(judge_name, str) or not judge_name.strip():
        return ""

    name = judge_name.strip()

    # Handle names with multiple commas like "Lastname, Firstname Middle, Jr."
    all_parts = name.split(",")

    # First part is always the last name
    last_name = all_parts[0].strip()

    # Check if there are multiple parts
    if len(all_parts) >= 2:
        # Last part might be a suffix (Jr., Sr., etc.)
        potential_suffix = all_parts[-1].strip().lower()

        # Check if the last part is a known suffix
        suffix_patterns = ["jr", "sr", "ii", "iii", "iv", "v"]
        is_suffix = any(
            potential_suffix == pattern or potential_suffix.startswith(pattern + ".")
            for pattern in suffix_patterns
        )

        if len(all_parts) > 2 and is_suffix:
            # If we have lastname, firstname middle, suffix format
            suffix = all_parts[-1].strip()
            first_and_middle = all_parts[
                1
            ].strip()  # Second part has firstname and possibly middle
        else:
            # If no suffix or just lastname, firstname middle format
            suffix = ""
            first_and_middle = ", ".join(all_parts[1:]).strip()  # Join everything after lastname

            # Check for embedded suffix in the first_and_middle part
            suffix_match = re.search(r"\b(Jr\.?|Sr\.?|I{1,3}V?|IV)\b", first_and_middle)
            if suffix_match:
                suffix = suffix_match.group(0)
                first_and_middle = re.sub(
                    r"\b(Jr\.?|Sr\.?|I{1,3}V?|IV)\b", "", first_and_middle
                ).strip()

        # Split first and middle names
        name_parts = first_and_middle.split()
        first_name = name_parts[0] if name_parts else ""
        middle_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

        # Reconstruct in desired format: first middle last suffix
        full_name = " ".join(filter(None, [first_name, middle_name, last_name, suffix]))
        return full_name
    else:
        # If the name doesn't contain a comma, return it as is
        return name


def extract_name_and_location_from_description(description: str) -> tuple[str, str]:
    """
    Extracts full name and location of origin from nomination description strings.

    The function parses description strings like:
    "melissa damian, of florida, to be ..."
    "nicole g. bernerr of maryland, to be united..."
    "sherri malloy beatty-arthur, of the district of columbia, for..."

    Args:
        description: A nomination description string

    Returns:
        Tuple of (full_name, location_of_origin)
        - full_name: The nominee's name (e.g., "melissa damian")
        - location_of_origin: The nominee's location (e.g., "florida")
    """
    if not isinstance(description, str) or not description.strip():
        return "", ""

    description = description.strip()

    # Pattern 1: "name, of location, to be/for..."
    pattern1 = r"^(.+?),\s+of\s+(.+?)(?:,\s+to\s+be|,\s+for|$)"

    # Pattern 2: "name, of the location, to be/for..."
    pattern2 = r"^(.+?),\s+of\s+the\s+(.+?)(?:,\s+to\s+be|,\s+for|$)"

    # Pattern 3: "name of location, to be/for..."
    pattern3 = r"^(.+?)\s+of\s+(.+?)(?:,\s+to\s+be|,\s+for|$)"

    # Try patterns in sequence
    for pattern in [pattern1, pattern2, pattern3]:
        match = re.search(pattern, description)
        if match:
            full_name = match.group(1).strip()
            location = match.group(2).strip().strip("the ")
            return full_name, location

    # If no match found, return empty strings
    logger.warning(f"Could not extract name and location from description: {description}")
    return "", ""


def extract_name_and_location_columns(
    df: pd.DataFrame, description_column: str = "description"
) -> pd.DataFrame:
    """
    Creates 'full_name_from_description' and 'location_of_origin_from_description'
    columns in the provided DataFrame by extracting from the description column.

    Args:
        df: DataFrame containing nomination descriptions
        description_column: Name of the column containing descriptions (default: "description")

    Returns:
        DataFrame with two new columns added: 'full_name_from_description' and
        'location_of_origin_from_description'
    """
    if description_column not in df.columns:
        logger.error(f"Column '{description_column}' not found in DataFrame")
        return df

    # Create a copy to avoid SettingWithCopyWarning
    result_df = df.copy()

    # Apply extraction function to each description
    extracted_data = result_df[description_column].apply(
        extract_name_and_location_from_description
    )

    # Split the returned tuples into separate columns
    result_df["full_name_from_description"] = extracted_data.apply(lambda x: x[0])
    result_df["location_of_origin_from_description"] = extracted_data.apply(lambda x: x[1])

    # Log extraction statistics
    name_count = result_df["full_name_from_description"].notna().sum()
    location_count = result_df["location_of_origin_from_description"].notna().sum()
    total_rows = len(result_df)

    logger.info(
        f"Extracted {name_count}/{total_rows} ({name_count / total_rows:.1%}) names and "
        f"{location_count}/{total_rows} ({location_count / total_rows:.1%}) locations"
    )

    return result_df

