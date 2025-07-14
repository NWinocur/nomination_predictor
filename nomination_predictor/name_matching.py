from loguru import logger
from nameparser import HumanName
import pandas as pd
import tqdm


def perform_exact_name_matching(
    congress_df: pd.DataFrame, 
    fjc_df: pd.DataFrame,
    congress_name_col: str = "full_name_from_description",
    fjc_name_col: str = "judge_name",
    nid_col: str = "nid"
) -> pd.DataFrame:
    """
    Performs exact string matching between Congress.gov and FJC names.
    
    Args:
        congress_df: DataFrame with Congress.gov nomination data
        fjc_df: DataFrame with FJC judge data
        congress_name_col: Column containing Congress nominee names
        fjc_name_col: Column containing FJC judge names
        nid_col: Column containing unique FJC identifier
        
    Returns:
        DataFrame with Congress records matched to FJC NIDs
    """
    logger.info(f"Starting exact name matching with {len(congress_df)} Congress records and {len(fjc_df)} FJC records")
    
    # Create result DataFrame to store matches
    results = []
    
    # Process each Congress record
    for idx, cong_row in tqdm.tqdm(congress_df.iterrows()):
        cong_name = cong_row.get(congress_name_col, "")
        
        # Skip if name is missing
        if not cong_name or pd.isna(cong_name):
            logger.debug(f"Skipping record {idx}: Missing name")
            continue
            
        # Parse name using nameparser
        parsed_cong_name = HumanName(cong_name)
        cong_last_name = parsed_cong_name.last.strip().casefold()
        cong_first_name = parsed_cong_name.first.strip().casefold()
        
        # Get middle name and middle initial if available
        cong_middle_name = ""
        cong_middle_initial = ""
        if parsed_cong_name.middle:
            cong_middle_name = parsed_cong_name.middle.strip().casefold()
            cong_middle_initial = cong_middle_name[0] if cong_middle_name else ""
        
        if not cong_last_name:
            logger.debug(f"Skipping record {idx}: Unable to parse last name from '{cong_name}'")
            continue
        
        # Find all FJC records with matching last name
        matches = []
        for _, fjc_row in fjc_df.iterrows():
            fjc_name = fjc_row.get(fjc_name_col, "")
            
            # Skip if name is missing
            if not fjc_name or pd.isna(fjc_name):
                continue
                
            # Parse name using nameparser
            parsed_fjc_name = HumanName(fjc_name)
            
            # Get middle name and middle initial if available
            fjc_middle_name = ""
            fjc_middle_initial = ""
            if parsed_fjc_name.middle:
                fjc_middle_name = parsed_fjc_name.middle.strip().casefold()
                fjc_middle_initial = fjc_middle_name[0] if fjc_middle_name else ""
            fjc_last_name = parsed_fjc_name.last.strip().casefold()
            fjc_first_name = parsed_fjc_name.first.strip().casefold()
            
            # Check for last name match (exact string comparison after casefolding and stripping)
            if cong_last_name == fjc_last_name:
                matches.append({
                    "cong_idx": idx,
                    "fjc_idx": _,
                    "nid": fjc_row.get(nid_col),
                    "fjc_name": fjc_name,
                    "cong_name": cong_name,
                    "first_name_match": cong_first_name == fjc_first_name,
                    "middle_name_match": (cong_middle_name == fjc_middle_name) if (cong_middle_name and fjc_middle_name) else None,
                    "parsed_cong_first": cong_first_name,
                    "parsed_fjc_first": fjc_first_name,
                    "parsed_cong_middle": cong_middle_name,
                    "parsed_fjc_middle": fjc_middle_name,
                    "parsed_cong_middle_initial": cong_middle_initial,
                    "parsed_fjc_middle_initial": fjc_middle_initial 
                })
        
        if not matches:
            logger.info(f"No last name matches found for '{cong_name}' (last: '{cong_last_name}')")
            continue
            
        # If only one row matched last name, check first name too (just in case it's a common last name)
            # If first name also matches, add it as being of match_type first_and_last_name and not ambiguous
            # Else add it to the results as being of match_type last_name_only and mark as ambiguous
        if len(matches) == 1:
            match = matches[0]
            
            # Check if first name also matches
            if match["first_name_match"]:
                results.append({
                    "congress_index": match["cong_idx"],
                    "congress_name": match["cong_name"],
                    "fjc_name": match["fjc_name"],
                    "nid": match["nid"],
                    "match_type": "first_and_last_name",
                    "ambiguous": False
                })
                logger.info(f"Exact match found for '{cong_name}' (first: '{cong_first_name}', last: '{cong_last_name}')")
            else:
                results.append({
                    "congress_index": match["cong_idx"],
                    "congress_name": match["cong_name"],
                    "fjc_name": match["fjc_name"],
                    "nid": match["nid"],
                    "match_type": "last_name_only",
                    "ambiguous": True
                })
                logger.info(f"Last name only match found for '{cong_name}' (last: '{cong_last_name}')")
            continue
            
        # If multiple matches, try to disambiguate by first name
        first_name_matches = [m for m in matches if m["first_name_match"]]
        
        if len(first_name_matches) == 1:
            match = first_name_matches[0]
            results.append({
                "congress_index": match["cong_idx"],
                "congress_name": match["cong_name"],
                "fjc_name": match["fjc_name"],
                "nid": match["nid"],
                "match_type": "first_and_last_name",
                "ambiguous": False
            })
            logger.info(f"Exact match found for '{cong_name}' (first: '{cong_first_name}', last: '{cong_last_name}')")

        elif len(first_name_matches) > 1:
            # if literally every entry uses same match['nid'], then each entry is really just a different service record for the same human being, so it can be treated as an unambiguous person-match
            # else if at least one of the matches uses a different match['nid'], then it's still ambiguous so far & we can go on to the "still ambiguous even with first name" path
            
            # Check if all matches have the same NID
            unique_nids = {match['nid'] for match in first_name_matches}
            if len(unique_nids) == 1:
                # All matches have the same NID - it's the same person with different service records
                match = first_name_matches[0]  # Use the first one as representative
                results.append({
                    "congress_index": match["cong_idx"],
                    "congress_name": match["cong_name"],
                    "fjc_name": match["fjc_name"],
                    "nid": match["nid"],
                    "match_type": "first_and_last_name_multiple_records",
                    "ambiguous": False
                })
                logger.info(f"Exact match found for '{cong_name}' (first: '{cong_first_name}', last: '{cong_last_name}')")
            else:
                # Try to disambiguate further using middle initial
                middle_initial_matches = [m for m in first_name_matches if m.get("middle_name_match")]
                
                if len(middle_initial_matches) == 1:
                    # If exactly one match has a matching middle initial, use it
                    match = middle_initial_matches[0]
                    results.append({
                        "congress_index": match["cong_idx"],
                        "congress_name": match["cong_name"],
                        "fjc_name": match["fjc_name"],
                        "nid": match["nid"],
                        "match_type": "first_middle_last_name",
                        "ambiguous": False
                    })
                    logger.info(f"Match found with first, middle initial, and last name for '{cong_name}'")
                else:
                    # Still ambiguous even with middle initial check
                    logger.info(f"Multiple matches even with first name for '{cong_name}':") 
                    for match in first_name_matches:
                        middle_match_str = ", middle: match" if match.get("middle_name_match") else ""
                        logger.info(f"  - {match['fjc_name']} (NID: {match['nid']}{middle_match_str})") 
                    
                    # Add all possible matches to results, marking as ambiguous
                    for match in first_name_matches:
                        results.append({
                            "congress_index": match["cong_idx"],
                            "congress_name": match["cong_name"],
                            "fjc_name": match["fjc_name"],
                            "nid": match["nid"],
                            "match_type": "first_and_last_name",
                            "ambiguous": True
                        })
        else:
            # No first name matches, but multiple last name matches
            logger.info(f"Multiple last name matches for '{cong_name}' (first: '{cong_first_name}'):")
            for match in matches:
                logger.info(f"  - {match['fjc_name']} (NID: {match['nid']}, first: '{match['parsed_fjc_first']}')")
                
            # Add all possible matches to results, marking as ambiguous
            for match in matches:
                results.append({
                    "congress_index": match["cong_idx"],
                    "congress_name": match["cong_name"],
                    "fjc_name": match["fjc_name"],
                    "nid": match["nid"],
                    "match_type": "last_name_only",
                    "ambiguous": True
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    logger.info(f"Exact name matching completed with {len(results_df)} total matches")
    logger.info(f"  - Unambiguous matches: {len(results_df[results_df['ambiguous'] == False])}")
    logger.info(f"  - Ambiguous matches: {len(results_df[results_df['ambiguous'] == True])}")
    
    return results_df


