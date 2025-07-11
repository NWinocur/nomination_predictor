# nomination_predictor/congress_api.py
"""Client for fetching judicial nomination data from Congress.gov API."""

from datetime import datetime
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
import requests


def is_likely_judicial_summary(nomination: Dict[str, Any]) -> bool:
    """
    Quick check on summary data to determine if we should fetch full details.
    This helps minimize API calls.
    """
    if not isinstance(nomination, dict):
        return False
        
    description = str(nomination.get("description", "")).lower()
    organization = str(nomination.get("organization", "")).lower()
    
    # Keywords that suggest a judicial position
    judicial_indicators = [
        # Court types
        'district court', 'circuit court', 'supreme court',
        'court of appeals', 'court of international trade',
        'tax court', 'claims court', 'court of federal claims',
        'veterans claims', 'bankruptcy court',
        # Position titles
        'judge', 'justice', 'magistrate',
        # Organization indicators
        'judicial', 'judiciary', 'court'
    ]
    
    return any(indicator in f"{description} {organization}" 
              for indicator in judicial_indicators)
    
def is_judicial_nomination(nomination_data: Dict[str, Any]) -> bool:
    """
    Determine if a nomination is for a judicial position.
    
    Handles both summary and detailed nomination responses from the Congress.gov API.
    
    Args:
        nomination_data: Nomination data from Congress.gov API, which could be:
            - A summary response (from get_nominations)
            - A detailed response (from get_nomination_detail)
        
    Returns:
        bool: True if this appears to be a judicial nomination, False otherwise
    """
    if not isinstance(nomination_data, dict):
        return False
    
    # Check top-level description
    description = str(nomination_data.get("description", "")).lower()
    
    # Check position title from top level or within nominees
    position_title = str(nomination_data.get("positionTitle", "")).lower()
    organization = str(nomination_data.get("organization", "")).lower()
    
    # Check nominees if available
    nominees = []
    if "nominees" in nomination_data:
        if isinstance(nomination_data["nominees"], dict) and "items" in nomination_data["nominees"]:
            nominees = nomination_data["nominees"]["items"]
        elif isinstance(nomination_data["nominees"], list):
            nominees = nomination_data["nominees"]
    elif "nominee" in nomination_data:  # Handle singular form if present
        nominees = [nomination_data["nominee"]]
    
    # Judicial indicators - ordered by specificity
    judicial_keywords = [
        # Court types
        'district court', 
        'circuit court', 
        'supreme court',
        'court of appeals',
        'international trade court',
        'tax court',
        'claims court',
        'veterans claims',
        # Position titles
        'judge',
        'justice',
        'magistrate',
        'bankruptcy judge',
        # Organization indicators
        'judicial',
        'judiciary'
    ]
    
    # Check description and top-level fields
    text_to_check = f"{description} {position_title} {organization}".lower()
    if any(keyword in text_to_check for keyword in judicial_keywords):
        return True
    
    # Check nominee-specific information
    for nominee in nominees:
        if not isinstance(nominee, dict):
            continue
            
        # Check nominee's position title and organization
        nom_position = str(nominee.get("positionTitle", "")).lower()
        nom_org = str(nominee.get("organization", "")).lower()
        nom_text = f"{nom_position} {nom_org}".lower()
        
        if any(keyword in nom_text for keyword in judicial_keywords):
            return True
    
    return False

def parse_court_from_description(description: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse circuit and court information from nomination description.
    
    Args:
        description: Nomination description text
        
    Returns:
        Tuple of (circuit number or None, court code or None)
    """
    # Try to extract circuit information
    circuit_match = re.search(r'(\d+)(?:st|nd|rd|th)? Circuit', description, re.IGNORECASE)
    if circuit_match:
        circuit = int(circuit_match.group(1))
    else:
        circuit = None
    
    # Try to extract district court information
    district_match = re.search(r'(Northern|Southern|Eastern|Western|Middle|Central) District of ([A-Z][a-z]+)', description, re.IGNORECASE)
    if district_match:
        direction = district_match.group(1)[0].upper()  # Get first letter (N, S, E, W, M, C)
        state_name = district_match.group(2)
        # Map state name to postal code (simplified example)
        state_mapping = {
            "California": "CA",
            "New York": "NY",
            "Florida": "FL",
            # Add more as needed
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
    
    # Process nominees
    nominees_data = nomination_data.get("nominees", {}).get("items", [])
    
    if not nominees_data and "description" in nomination_data:
        # If no detailed nominee data available, create a single record from description
        description = nomination_data.get("description", "")
        base_record["description"] = description
        
        # Try to extract nominee name from description
        name_match = re.search(r'^([^,]+),', description)
        if name_match:
            base_record["nominee"] = name_match.group(1).strip()
        
        # Try to extract circuit/court from description
        circuit, court = parse_court_from_description(description)
        if circuit is not None:
            base_record["circuit"] = circuit
        if court is not None:
            base_record["court"] = court
        
        records.append(base_record.copy())
    else:
        # Process each nominee
        for nominee in nominees_data:
            record = base_record.copy()
            
            # Nominee information
            record["nominee"] = " ".join(filter(None, [
                nominee.get("firstName", ""),
                nominee.get("middleName", ""),
                nominee.get("lastName", "")
            ]))
            record["state"] = nominee.get("state")
            
            # Position information
            record["position"] = nominee.get("positionTitle", "")
            record["organization"] = nominee.get("organization", "")
            
            # Extract circuit/court information from position title
            circuit, court = parse_court_from_description(record["position"])
            if circuit is not None:
                record["circuit"] = circuit
            if court is not None:
                record["court"] = court
            
            records.append(record)
    
    return records


class CongressAPIClient:
    """Client for interacting with Congress.gov API."""
    
    BASE_URL = "https://api.congress.gov/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("CONGRESS_API_KEY")
        if not self.api_key:
            raise ValueError("Congress.gov API key is required")
            
    def get_nominations(self, congress: int, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch nominations for a specific congress."""
        url = f"{self.BASE_URL}/nomination/{congress}"
        default_params = {"api_key": self.api_key, "format": "json", "limit": 250}
        if params:
            default_params.update(params)
        
        logger.info(f"Fetching nominations for {congress}th Congress")
        response = requests.get(url, params=default_params)
        response.raise_for_status()
        return response.json()

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
        response = requests.get(url_path, params=params)
        response.raise_for_status()
        return response.json()

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

        Args:
            congress: Congress number (e.g., 118)

        Returns:
            List of judicial nominations in the project's format
        """
        records = []
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

        # First pass: Filter likely judicial nominations
        potential_judicial = [
            nom for nom in nominations
            if isinstance(nom, dict) and "number" in nom and is_likely_judicial_summary(nom)
        ]

        logger.info(f"Found {len(potential_judicial)} potential judicial nominations")

        # Second pass: Get details and verify
        for i, nom in enumerate(potential_judicial, 1):
            try:
                nomination_number = nom["number"]
                logger.debug(f"Processing potential judicial nomination {i}/{len(potential_judicial)}: {nomination_number}")

                # Get detailed information
                detail = self.get_nomination_detail(congress, nomination_number)
                if not detail:
                    continue

                # Verify with full details
                if not is_judicial_nomination(detail):
                    logger.debug(f"Excluded after detail check: {detail.get('description', 'Unknown')}")
                    continue

                # Transform and add to records
                transformed = transform_nomination_data(detail)
                if transformed:
                    records.extend(transformed)
                    logger.info(f"Added {len(transformed)} records for judicial nomination {nomination_number}")

            except Exception as e:
                logger.error(f"Error processing nomination {nomination_number}: {str(e)}", exc_info=True)
                continue

        logger.info(f"Completed processing. Found {len(records)} judicial nominations")
        return records