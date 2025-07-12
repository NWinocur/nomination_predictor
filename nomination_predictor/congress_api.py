"""Client for fetching judicial nomination data from Congress.gov API."""

from datetime import datetime
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from loguru import logger
import requests

# Tokens that strongly indicate a nomination is judicial when found in organization field
JUDICIAL_ORG_TOKENS = [
    "the judiciary",
    "judicial",
    "courts",
    "court of",
    "supreme court",
    "district court",
    "circuit court",
    "bankruptcy court",
    "federal court",
    "court of federal",
    "court of appeals",
    "court of claims",
    "court of international",
    "u.s. court"
]

# Tokens that may indicate a judicial nomination when found in description/position/text fields
JUDICIAL_DESC_TOKENS = [
    "judge",
    "justice",
    "circuit",
    "district",
    "bankruptcy",
    "judicial",
    "judiciary",
    "court",
    "magistrate"
]


def is_likely_judicial_summary(nom: dict) -> bool:
    """
    First-level filter to identify likely judicial nominations from summary data.
    Designed to be permissive to avoid false negatives.
    
    Args:
        nom: Nomination summary data
        
    Returns:
        True if nomination is likely a judicial nomination
    """
    # Check description for judicial keywords
    description = nom.get("description", "").lower()
    return any(token.lower() in description for token in JUDICIAL_DESC_TOKENS)


def is_judicial_nomination(nom: dict) -> bool:
    """
    Determine if a nomination is for a judicial position based on the nomination data.
    
    This function checks various fields in the nomination data to see if it's a judicial
    nomination. It's designed to work with both summary and detail responses.
    
    Args:
        nom: Nomination data from Congress.gov API
        
    Returns:
        True if the nomination is for a judicial position, False otherwise
    """
    # Extract text fields to search for judicial keywords
    description = nom.get("description", "").lower()
    
    # Check for nominees and their positions
    nominees_data = nom.get("nominees", {})
    if isinstance(nominees_data, dict) and "items" in nominees_data:
        nominees = nominees_data.get("items", [])
    else:
        nominees = []
    
    # Extract organizations and position titles from nominees
    orgs = []
    position_titles = []
    for nominee in nominees:
        if isinstance(nominee, dict):
            org = nominee.get("organization", "").lower()
            if org:
                orgs.append(org)
            
            position = nominee.get("positionTitle", "").lower()
            if position:
                position_titles.append(position)
    
    # Check latest action
    latest_action = nom.get("latestAction", {})
    latest_action_text = latest_action.get("text", "").lower() if latest_action else ""
    
    # Check for judicial organization first (most reliable)
    for org in orgs:
        if any(token.lower() in org for token in JUDICIAL_ORG_TOKENS):
            return True
    
    # Check position titles and other text fields for judicial keywords
    search_text = " ".join([description, latest_action_text] + position_titles)
    
    # Now check for judicial keywords in combined text
    return any(token.lower() in search_text for token in JUDICIAL_DESC_TOKENS)


def parse_court_from_description(description: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse circuit and court information from nomination description.
    
    Args:
        description: Nomination description text
        
    Returns:
        Tuple of (circuit number or None, court code or None)
    """
    circuit = None
    court = None
    
    # Common patterns in judicial nominations
    circuit_pattern = r'(?:(\d+)(?:st|nd|rd|th)?\s+Circuit)'
    district_pattern = r'(?:District\s+of\s+([A-Za-z\s]+))'
    
    # Try to extract circuit
    circuit_match = re.search(circuit_pattern, description, re.IGNORECASE)
    if circuit_match:
        circuit = circuit_match.group(1).zfill(2)  # Pad with zero if needed
    
    # Try to extract district
    district_match = re.search(district_pattern, description, re.IGNORECASE)
    if district_match:
        state_name = district_match.group(1).strip()
        
        # Extract directional suffix if present (e.g., "Northern", "Eastern")
        direction = None
        for dir_match in re.finditer(r'\b(Northern|Southern|Eastern|Western|Central|Middle)\b', description, re.IGNORECASE):
            direction = dir_match.group(1)[0].upper()  # Just take first letter
            break
        
        # Map state name to code
        state_mapping = {
            "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
            "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
            "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
            "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
            "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
            "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
            "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
            "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
            "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
            "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
            "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
            "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
            "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
            "Puerto Rico": "PR", "Virgin Islands": "VI", "Guam": "GU",
            "American Samoa": "AS", "Northern Mariana Islands": "MP"
        }
        state_code = state_mapping.get(state_name)
        if state_code:
            court = f"{state_code}-{direction}"
        else:
            court = None
    else:
        court = None
    
    return circuit, court


def normalize_column_names(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize column names in API data (similar to how we handle FJC data).
    
    Args:
        data: Dictionary with API data
        
    Returns:
        Dictionary with normalized column names (lowercase, underscores)
    """
    if not isinstance(data, dict):
        return data
        
    if not data:
        return {}
        
    # Convert camelCase or PascalCase to snake_case
    normalized_data = {}
    for key, value in data.items():
        # Convert to lowercase
        new_key = key.lower()
        # Replace spaces with underscores
        new_key = new_key.replace(" ", "_")
        normalized_data[new_key] = value
        
    return normalized_data


def extract_nominee_data(nominee_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract nominee-specific data from Congress.gov API nominee endpoint response,
    preserving original values and structure. Only normalizes column names.
    
    Args:
        nominee_data: Nominee data from Congress.gov API
        
    Returns:
        Dictionary with nominee data and minimal transformations
    """
    logger.debug("Extracting nominee data from API response")
    
    # Create a record with original API data values
    record = {}
    
    # Copy all non-nested fields directly
    for key, value in nominee_data.items():
        if not isinstance(value, (dict, list)) or key == 'nominees':
            record[key] = value
    
    # Add citation as a correlatable field
    if 'congress' in nominee_data and 'number' in nominee_data:
        congress = nominee_data.get('congress')
        number = nominee_data.get('number')
        ordinal = nominee_data.get('ordinal', '1')
        record['citation'] = f"PN{number}"
        record['nominee_id'] = f"{congress}-{number}-{ordinal}"
    
    # Add source metadata
    record['data_source'] = 'congress.gov_api_nominee'
    record['retrieval_date'] = datetime.now().isoformat()
    
    return record


def extract_nomination_data(nomination_data: Dict[str, Any], full_details: bool = False) -> List[Dict[str, Any]]:
    """
    Extract Congress.gov API nomination data, preserving original values and structure.
    Only normalizes column names for consistency with other data sources.
    
    Args:
        nomination_data: Nomination data from Congress.gov API
        full_details: Whether this is a full nomination detail object (to properly trace logs)
        
    Returns:
        List of nomination records with minimal transformations
    """
    logger.debug(f"Extracting nomination data (full_details={full_details})")
    logger.trace(f"Input data: {nomination_data}")
    
    records = []
    
    # Add metadata fields that aren't in the API response
    metadata = {
        "data_source": "congress.gov_api",  # Track the data source
        "retrieval_date": datetime.now().isoformat(),  # When this data was retrieved
        "is_detail_record": full_details,  # Whether this came from detail or summary
    }
    
    # Process nominees - handle both list and dict formats
    nominees_data = []
    
    # Get nominees, handling both dictionary and list formats
    nominees = nomination_data.get("nominees", {})
    if isinstance(nominees, dict) and "items" in nominees:
        # Structure: {"nominees": {"items": [...]}}
        nominees_data = nominees["items"]
    elif isinstance(nominees, list):
        # Structure: {"nominees": [...]}
        nominees_data = nominees
    
    # If no nominees data available, create a single record with just nomination data
    if not nominees_data:
        # Create a base record with all top-level nomination data
        record = {}
        
        # Copy all top-level fields from the nomination data
        for key, value in nomination_data.items():
            if key != "nominees":  # Skip nominees field since we're handling separately
                record[key] = value
                
        # Add metadata
        record.update(metadata)
        records.append(record)
    else:
        # Process each nominee separately
        for nominee in nominees_data:
            record = {}
            
            # Copy all top-level fields from the nomination data
            for key, value in nomination_data.items():
                if key != "nominees":  # Skip nominees field since we're handling separately
                    record[key] = value
            
            # Add all nominee fields with nominee_ prefix to avoid collisions
            for key, value in nominee.items():
                record[f"nominee_{key}"] = value
                
            # Add metadata
            record.update(metadata)
            records.append(record)
    
    return records


class CongressAPIClient:
    """Client for interacting with Congress.gov API."""
    
    BASE_URL = "https://api.congress.gov/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("CONGRESS_API_KEY")
        if not self.api_key:
            raise ValueError("Congress.gov API key is required")
            
    def get_nominations(self, congress: int, params: Dict[str, Any] = None, auto_paginate:bool=True) -> Dict[str, Any]:
        """
        Fetch nominations for a specific congress with pagination support.
        
        This method automatically handles pagination by following the 'next' link
        in the API response until all records are retrieved.
        
        Args:
            congress: Congress number (e.g., 118)
            params: Additional parameters to pass to the API
            
        Returns:
            Combined results from all pages with 'nominations' key containing
            all nominations across all pages
        """
        url = f"{self.BASE_URL}/nomination/{congress}"
        default_params = {"api_key": self.api_key, "format": "json", "limit": 250}  # 250 is server-enforced limit; tried 500 and still got 250
        if params:
            default_params.update(params)
        
        logger.info(f"Fetching nominations for {congress}th Congress with auto-pagination option set to {auto_paginate}")
        
        # Initialize result structure
        combined_result = None
        page = 1
        total_nominations = []
        
        while True:
            logger.info(f"Fetching page {page} for {congress}th Congress nominations")
            response = requests.get(url, params=default_params)
            response.raise_for_status()
            current_page = response.json()
            
            # Initialize the combined result with the first page
            if combined_result is None:
                combined_result = current_page
                if "nominations" not in combined_result:
                    logger.warning("No 'nominations' key found in API response")
                    return combined_result
                    
            # Extract nominations from current page
            current_nominations = current_page.get("nominations", [])
            total_nominations.extend(current_nominations)
            
            logger.info(f"Retrieved {len(current_nominations)} nominations from page {page}")
            
            # Check if there's a next page link
            if "pagination" in current_page and "next" in current_page["pagination"] and auto_paginate:
                # Extract the offset for the next page
                next_link = current_page["pagination"]["next"]
                if not next_link:
                    logger.debug("No more pages available")
                    break
                
                try:
                    # Parse the next link to get the offset parameter
                    # Handle different URL formats with or without '?' or '&'
                    parsed_url = urlparse(next_link)
                    query_params = parse_qs(parsed_url.query)
                    
                    if "offset" in query_params:
                        offset = query_params["offset"][0]
                        default_params["offset"] = offset
                        page += 1
                        logger.info(f"Moving to page {page} with offset {offset}")
                    else:
                        logger.warning(f"No offset parameter in next link: {next_link}")
                        break
                except Exception as e:
                    logger.warning(f"Error parsing next link '{next_link}': {str(e)}")
                    break
            else:
                logger.debug("No pagination information or next link, ending pagination")
                break
        
        # Replace the nominations in the first page result with all nominations
        if combined_result and "nominations" in combined_result:
            combined_result["nominations"] = total_nominations
            logger.info(f"Total nominations retrieved after pagination: {len(total_nominations)}")
        
        return combined_result

    def get_nomination_detail(self, congress: int, nomination_number: int, part_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch detailed information about a specific nomination.
        
        Args:
            congress: Congress number (e.g., 118)
            nomination_number: Nomination number
            part_number: Optional part number for partitioned nominations
            
        Returns:
            Detailed nomination data
        """
        url = f"{self.BASE_URL}/nomination/{congress}/{nomination_number}"
        params = {"api_key": self.api_key, "format": "json"}
        
        if part_number:
            url += f"/{part_number}"
            
        logger.info(f"Fetching nomination detail for {congress}-{nomination_number}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Nomination detail may be nested under 'nomination' key
        if "nomination" in data:
            nomination = data["nomination"]
            
            # The API sometimes returns a list instead of a dict for the 'nomination' key
            if isinstance(nomination, list) and len(nomination) > 0:
                nomination = nomination[0]  # Take the first item
                
            return nomination
        
        return data
        
    def get_nomination_nominees(self, congress: int, nomination_number: int) -> List[Dict[str, Any]]:
        """
        Extract nominee information from the nomination detail response.
        
        According to the API documentation, nominee data is included within the main nomination
        detail response, not at a separate /nominees endpoint. The structure is:
        
        - <nomination>
          - <nominees>
            - <item>
              - <firstName>
              - <lastName>
              - <middleName>
              - <prefix>
              - <suffix>
              - <state>
              - etc.
        
        Args:
            congress: Congress number (e.g., 118)
            nomination_number: Nomination number
            
        Returns:
            List of nominees with detailed information
        """
        logger.info(f"Extracting nominee information for {congress}-{nomination_number}")
        
        try:
            # Get the nomination detail which contains nominee information
            detail = self.get_nomination_detail(congress, nomination_number)
            
            # Extract nominees from the detail response
            if not detail:
                logger.warning(f"No nomination detail found for {congress}-{nomination_number}")
                return []
                
            # Check for nominees in the detail response
            if "nominees" in detail:
                nominees = detail["nominees"]
                
                # Handle different possible structures for nominees
                if isinstance(nominees, list):
                    return nominees
                elif isinstance(nominees, dict):
                    if "item" in nominees:
                        # Format: {"nominees": {"item": [...]}}
                        items = nominees["item"]
                        return items if isinstance(items, list) else [items]
                    else:
                        # Single nominee
                        return [nominees]
            else:
                logger.warning(f"No nominees field in nomination detail for {congress}-{nomination_number}")
                # For nominations with a single nominee, the nominee info might be directly in the nomination
                # Try to extract nominee info from the nomination itself as fallback
                nominee_fields = ["firstName", "lastName", "organization", "positionTitle"]
                if any(field in detail for field in nominee_fields):
                    logger.info(f"Using nomination detail as nominee for {congress}-{nomination_number}")
                    return [detail]
                
            return []
        except Exception as e:
            logger.error(f"Error extracting nominee data for {congress}-{nomination_number}: {str(e)}")
            return []
            
    def get_nomination_actions(self, congress: int, nomination_number: int) -> Dict[str, Any]:
        """
        Fetch actions for a specific nomination.
        
        Args:
            congress: Congress number (e.g., 118)
            nomination_number: Nomination number
            
        Returns:
            Nomination actions data
        """
        url = f"{self.BASE_URL}/nomination/{congress}/{nomination_number}/actions"  # the /actions endpoint does exist, according to documentation
        params = {"api_key": self.api_key, "format": "json"}
        
        logger.info(f"Fetching actions for nomination {nomination_number} ({congress}th Congress)")
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
        
    def get_nominee_data_from_url(self, nominee_url: str) -> Dict[str, Any]:
        """
        Fetch nominee-specific data from a nominee URL.
        
        Args:
            nominee_url: URL to fetch nominee data from (e.g., https://api.congress.gov/v3/nomination/118/2012/1?format=json)
            
        Returns:
            Dictionary with nominee data
        """
        logger.info(f"Fetching nominee data from URL: {nominee_url}")
        
        # Parse the URL to extract congress, nomination number, and ordinal
        try:
            # Extract path components to identify congress, nomination number, and ordinal
            path_parts = urlparse(nominee_url).path.split('/')
            # Format should be /v3/nomination/{congress}/{nomination_number}/{ordinal}
            if len(path_parts) >= 6 and path_parts[1] == "v3" and path_parts[2] == "nomination":
                congress = path_parts[3]
                nomination_number = path_parts[4]
                ordinal = path_parts[5]
                logger.debug(f"Parsed nominee URL: congress={congress}, nomination_number={nomination_number}, ordinal={ordinal}")
            else:
                logger.warning(f"Nominee URL has unexpected format: {nominee_url}")
                return {}
        except Exception as e:
            logger.error(f"Error parsing nominee URL {nominee_url}: {str(e)}")
            return {}
        
        # Construct API URL with authorization
        params = {"api_key": self.api_key, "format": "json"}
        
        try:
            # Make the API request
            response = requests.get(nominee_url.split('?')[0], params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract nominee data - could be under different keys depending on endpoint structure
            nominee_data = {}
            if "nominee" in data:
                nominee_data = data["nominee"]
            elif "nominees" in data:
                # Handle different nominee structures
                nominees = data["nominees"]
                if isinstance(nominees, list) and nominees:
                    nominee_data = nominees[0]
                elif isinstance(nominees, dict):
                    if "item" in nominees:
                        items = nominees["item"]
                        if isinstance(items, list) and items:
                            nominee_data = items[0]
                        elif isinstance(items, dict):
                            nominee_data = items
                    else:
                        nominee_data = nominees
            else:
                # If no specific nominee key, use the whole response
                nominee_data = data
            
            # Add context information
            nominee_data['congress'] = congress
            nominee_data['number'] = nomination_number
            nominee_data['ordinal'] = ordinal
            nominee_data['nominee_url'] = nominee_url
            
            return nominee_data
        except Exception as e:
            logger.error(f"Error fetching nominee data from {nominee_url}: {str(e)}")
            return {}
    
    def get_all_nominees_data(self, nominations_df) -> List[Dict[str, Any]]:
        """
        Fetch nominee data for all nominations in a DataFrame that have nominee_url.
        
        Args:
            nominations_df: DataFrame with nomination data including nominee_url field
            
        Returns:
            List of nominee data dictionaries
        """
        logger.info(f"Fetching nominee data for {len(nominations_df)} nominations")
        
        nominees_data = []
        
        for i, row in enumerate(nominations_df.to_dict('records')):
            if 'nominee_url' in row and row['nominee_url']:
                try:
                    logger.info(f"Processing nominee {i+1}/{len(nominations_df)}: {row.get('citation', 'Unknown')}")
                    nominee_data = self.get_nominee_data_from_url(row['nominee_url'])
                    
                    if nominee_data:
                        # Extract and normalize nominee data
                        nominee_record = extract_nominee_data(nominee_data)
                        
                        # Add correlation fields from the nomination
                        if 'citation' in row and 'citation' not in nominee_record:
                            nominee_record['nomination_citation'] = row['citation']
                        
                        nominees_data.append(nominee_record)
                        logger.info(f"Added nominee data for {row.get('citation', 'Unknown')}")
                    else:
                        logger.warning(f"No data retrieved for nominee URL: {row['nominee_url']}")
                except Exception as e:
                    logger.error(f"Error processing nominee {row.get('citation', 'Unknown')}: {str(e)}")
            else:
                logger.debug(f"No nominee_url for nomination {row.get('citation', 'Unknown')}")
        
        logger.info(f"Retrieved data for {len(nominees_data)} nominees")
        return nominees_data

    def get_judicial_nominations(self, congress: int, auto_paginate:bool=True) -> List[Dict[str, Any]]:
        """
        Fetch and filter for judicial nominations from a single congress.
        
        Uses only summary-level filtering to identify judicial nominations,
        then retrieves detailed data for each one that passes the filter.
        Returns raw API data with minimal processing (column normalization only).

        Args:
            congress: Congress number (e.g., 118)

        Returns:
            List of judicial nomination records with minimal processing
        """
        logger.info(f"Fetching judicial nominations for Congress {congress}")
        
        # First get all civilian nominations (exclude military)
        all_nominations = self.get_nominations(congress, {"isCivilian": "true"}, auto_paginate=auto_paginate)    
        
        if not all_nominations or "nominations" not in all_nominations:
            logger.warning(f"No nominations found for Congress {congress}")
            return []
        
        # Get the actual nomination items from the response
        nominations = all_nominations.get("nominations", [])
        logger.info(f"Found {len(nominations)} civilian nominations in Congress {congress}")
        
        # Filter for likely judicial nominations using summary data
        judicial_nominations = [n for n in nominations if is_likely_judicial_summary(n)]
        logger.info(f"Found {len(judicial_nominations)} judicial nominations based on summary data")
        
        # Log some examples of what we found
        for i, nom in enumerate(judicial_nominations[:10]):
            logger.info(f"Judicial nomination {i+1}: {nom.get('number')} - {nom.get('description', 'No description')}")
        
        # For each judicial nomination, get the detailed information
        detailed_records = []
        
        for i, nomination in enumerate(judicial_nominations):
            congress_number = nomination.get("congress")
            nomination_number = nomination.get("number")
            
            if not congress_number or not nomination_number:
                logger.warning(f"Missing congress or nomination number for entry {i}")
                continue
                
            logger.info(f"Processing judicial nomination {i+1}/{len(judicial_nominations)}: {nomination_number}")
            
            # First extract records from summary data as fallback
            summary_records = extract_nomination_data(nomination, full_details=False)
            
            # Fetch detailed information
            try:
                # Get detailed nomination data - already contains nominee information
                detail = self.get_nomination_detail(congress_number, nomination_number)
                
                if not detail:
                    logger.warning(f"No detail found for nomination {nomination_number}, using summary data")
                    detailed_records.extend(summary_records)
                    continue
                    
                # Extract the detail data with minimal processing
                records = extract_nomination_data(detail, full_details=True)
                
                if records:
                    detailed_records.extend(records)
                    logger.info(f"Added {len(records)} detail-level records for nomination {nomination_number}")
                else:
                    # If no detailed records, use the summary records
                    logger.warning(f"No detail records created for {nomination_number}, falling back to summary")
                    detailed_records.extend(summary_records)
                    logger.info(f"Added {len(summary_records)} summary-level records for nomination {nomination_number}")
            except Exception as e:
                logger.error(f"Error processing nomination {nomination_number}: {str(e)}")
                # Fall back to summary records on error
                detailed_records.extend(summary_records)
                logger.info(f"Added {len(summary_records)} summary-level records for nomination {nomination_number} (after error)")
                # Print traceback for easier debugging
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Apply column name normalization to all records
        normalized_records = [normalize_column_names(record) for record in detailed_records]
        
        logger.info(f"Processed {len(normalized_records)} total judicial nomination records for Congress {congress}")
        return normalized_records
