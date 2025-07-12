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


def transform_nomination_data(nomination_data: Dict[str, Any], full_details: bool = False) -> List[Dict[str, Any]]:
    """
    Transform Congress.gov API nomination data into project's record format.
    
    Args:
        nomination_data: Nomination data from Congress.gov API
        full_details: Whether this is a full nomination detail object
        
    Returns:
        List of records in the project's format
    """
    logger.debug(f"Transforming nomination data (full_details={full_details})")
    logger.trace(f"Input data: {nomination_data}")
    
    records = []
    
    # Basic nomination information
    base_record = {
        "congress": nomination_data.get("congress"),
        "nomination_number": nomination_data.get("number"),
        "citation": nomination_data.get("citation"),
        "source": "congress.gov_api",
        "source_year": datetime.now().year,  # Current data fetch year
        "source_month": datetime.now().month,  # Current data fetch month
    }
    
    # Parse dates
    received_date = nomination_data.get("receivedDate")
    if received_date:
        base_record["nomination_date"] = received_date
    
    # Extract latest action
    latest_action = nomination_data.get("latestAction", {})
    if latest_action:
        base_record["latest_action_date"] = latest_action.get("actionDate")
        base_record["latest_action_text"] = latest_action.get("text")
    
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
    
    # Check if nominees_data contains actual nominee objects with first/last names
    has_valid_nominees = False
    for nominee in nominees_data:
        if isinstance(nominee, dict) and (nominee.get("firstName") or nominee.get("lastName")):
            has_valid_nominees = True
            break
    
    # Process from description if no valid nominees found
    if (not nominees_data or not has_valid_nominees) and "description" in nomination_data:
        # If no detailed nominee data available, create a single record from description
        description = nomination_data.get("description", "")
        base_record["description"] = description
        
        # Try to extract nominee name from description
        name_match = re.search(r'^([^,]+),', description)
        if name_match:
            base_record["nominee"] = name_match.group(1).strip()
        
        # Try to extract circuit/court from description
        circuit, court = parse_court_from_description(description)
        
        if circuit:
            base_record["circuit"] = circuit
            
        if court:
            base_record["court"] = court
        
        records.append(base_record)
    
    else:
        # Process each nominee separately
        for nominee in nominees_data:
            nominee_record = base_record.copy()
            
            # Basic nominee info
            first_name = nominee.get("firstName", "")
            last_name = nominee.get("lastName", "")
            
            # If we have a name from the nominee object, use it
            if first_name or last_name:
                nominee_record["nominee"] = f"{first_name} {last_name}".strip()
            else:
                # Try to extract from description as fallback
                description = nomination_data.get("description", "")
                name_match = re.search(r'^([^,]+),', description)
                if name_match:
                    nominee_record["nominee"] = name_match.group(1).strip()
            
            # Position info
            nominee_record["position_title"] = nominee.get("positionTitle", "")
            nominee_record["organization"] = nominee.get("organization", "")
            
            # Additional metadata
            description = nomination_data.get("description", "")
            nominee_record["description"] = description
            
            # Try to extract circuit/court from description or position
            circuit, court = parse_court_from_description(description)
            
            if not circuit and nominee_record.get("position_title"):
                # Try from position if not found in description
                position_circuit, position_court = parse_court_from_description(
                    nominee_record["position_title"]
                )
                if position_circuit:
                    circuit = position_circuit
                if position_court:
                    court = position_court
            
            if circuit:
                nominee_record["circuit"] = circuit
                
            if court:
                nominee_record["court"] = court
            
            records.append(nominee_record)
    
    return records


class CongressAPIClient:
    """Client for interacting with Congress.gov API."""
    
    BASE_URL = "https://api.congress.gov/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("CONGRESS_API_KEY")
        if not self.api_key:
            raise ValueError("Congress.gov API key is required")
            
    def get_nominations(self, congress: int, params: Dict[str, Any] = None) -> Dict[str, Any]:
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
        default_params = {"api_key": self.api_key, "format": "json", "limit": 250}  # Set to known working limit
        if params:
            default_params.update(params)
        
        logger.info(f"Fetching nominations for {congress}th Congress with pagination")
        
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
            if "pagination" in current_page and "next" in current_page["pagination"]:
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
        url_path = f"{self.BASE_URL}/nomination/{congress}/{nomination_number}"
        if part_number:
            url_path += f"/{part_number}"
            
        params = {"api_key": self.api_key, "format": "json"}
        
        logger.info(f"Fetching detail for nomination {nomination_number} ({congress}th Congress)")
        try:
            response = requests.get(url_path, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract the nomination from the nested response - handle both list and dict formats
            nomination_data = {}
            
            if isinstance(data, dict) and 'nomination' in data:
                # Handle both list and dictionary response formats
                if isinstance(data['nomination'], list) and data['nomination']:
                    # If it's a list, use the first item
                    logger.debug(f"Nomination data returned as a list with {len(data['nomination'])} items")
                    nomination_data = data['nomination'][0]
                elif isinstance(data['nomination'], dict):
                    # If it's a dictionary, use it directly
                    nomination_data = data['nomination']
                else:
                    logger.warning(f"Unexpected nomination format: {type(data['nomination'])}")
            else:
                logger.warning("No 'nomination' key found in response or response is not a dictionary")
                logger.debug(f"Response structure: {type(data)}")
                # Try to salvage by returning the entire response if it's a dict
                if isinstance(data, dict):
                    nomination_data = data
            
            # Log basic info about the response
            logger.debug(f"Received detail for nomination {nomination_number}")
            
            # Log important fields for debugging
            if isinstance(nomination_data, dict) and 'description' in nomination_data:
                logger.debug(f"Nomination description: {nomination_data['description']}")
            
            # Handle nominees data structure correctly
            if isinstance(nomination_data, dict):
                nominees_data = nomination_data.get('nominees', {})
                if isinstance(nominees_data, dict) and 'items' in nominees_data:
                    nominees = nominees_data['items']
                    logger.debug(f"Found {len(nominees)} nominees")
                    for i, nominee in enumerate(nominees, 1):
                        logger.debug(f"  Nominee {i}: {nominee.get('firstName', '')} {nominee.get('lastName', '')} - {nominee.get('positionTitle', 'No position')}")
            
            return nomination_data
            
        except Exception as e:
            logger.error(f"Error fetching detail for nomination {nomination_number}: {str(e)}")
            logger.debug(f"URL: {url_path}")
            logger.debug(f"Params: {params}")
            if 'response' in locals() and hasattr(response, 'text'):
                logger.debug(f"Response text: {response.text[:500]}...")
            raise

    def get_nomination_actions(self, congress: int, nomination_number: int) -> Dict[str, Any]:
        """
        Fetch actions for a specific nomination.
        
        Args:
            congress: Congress number (e.g., 118)
            nomination_number: Nomination number
            
        Returns:
            Nomination actions data
        """
        url = f"{self.BASE_URL}/nomination/{congress}/{nomination_number}/actions"
        params = {"api_key": self.api_key, "format": "json"}
        
        logger.info(f"Fetching actions for nomination {nomination_number} ({congress}th Congress)")
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
        
    def get_judicial_nominations(self, congress: int) -> List[Dict[str, Any]]:
        """
        Fetch and filter for judicial nominations from a single congress.
        
        Uses only summary-level filtering to identify judicial nominations,
        then retrieves detailed data for each one that passes the filter.

        Args:
            congress: Congress number (e.g., 118)

        Returns:
            List of judicial nominations in the project's format
        """
        records: list[dict[str, Any]] = []
        logger.info(f"Fetching judicial nominations for Congress {congress}")

        params = {
            "api_key": self.api_key,
            "format": "json",
            "nominationType": "Civilian"
        }

        # Get the summary list of nominations
        response = self.get_nominations(congress, params=params)
        if not response or not isinstance(response, dict):
            logger.warning(f"Unexpected response format for Congress {congress}")
            return records

        nominations = response.get("nominations", [])
        logger.info(f"Found {len(nominations)} civilian nominations in Congress {congress}")

        # Filter judicial nominations using only summary-level filter
        judicial_nominations = [
            nom for nom in nominations
            if isinstance(nom, dict) and "number" in nom and is_likely_judicial_summary(nom)
        ]

        logger.info(f"Found {len(judicial_nominations)} judicial nominations based on summary data")
        
        # Log the first few judicial nominations identified
        for i, nom in enumerate(judicial_nominations[:3], 1):
            logger.info(f"Judicial nomination {i}: {nom.get('number')} - {nom.get('description', 'Unknown')}")

        # Process each judicial nomination
        for i, nom in enumerate(judicial_nominations, 1):
            try:
                nomination_number = nom["number"]
                logger.info(f"Processing judicial nomination {i}/{len(judicial_nominations)}: {nomination_number}")
                
                # First, create records from summary data as fallback
                summary_records = transform_nomination_data(nom)
                
                try:
                    # Get detailed information 
                    detail = self.get_nomination_detail(congress, nomination_number)
                    if detail:
                        # Transform the detail-level data (which has more information)
                        detail_records = transform_nomination_data(detail, full_details=True)
                        
                        if detail_records:
                            # Use the detail records which have more complete information
                            records.extend(detail_records)
                            logger.info(f"Added {len(detail_records)} detail-level records for nomination {nomination_number}")
                        else:
                            # Fall back to summary records if detail transformation fails
                            records.extend(summary_records)
                            logger.info(f"Added {len(summary_records)} summary-level records for nomination {nomination_number}")
                    else:
                        # Fall back to summary records if detail retrieval fails
                        records.extend(summary_records)
                        logger.info(f"Added {len(summary_records)} summary-level records for nomination {nomination_number} (detail unavailable)")
                        
                except Exception as e:
                    logger.error(f"Error retrieving details for nomination {nomination_number}: {str(e)}")
                    # Fall back to summary records on error
                    records.extend(summary_records)
                    logger.info(f"Added {len(summary_records)} summary-level records for nomination {nomination_number} (detail error)")

            except Exception as e:
                logger.error(f"Error processing nomination {nomination_number}: {str(e)}", exc_info=True)
                continue

        logger.info(f"Completed processing. Found {len(records)} judicial nomination records")
        return records
