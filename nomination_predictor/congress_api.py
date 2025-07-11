# nomination_predictor/congress_api.py
"""Client for fetching judicial nomination data from Congress.gov API."""

from datetime import datetime
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
import requests

JUDICIAL_ORG_TOKENS = [
    "the judiciary",                      # Article III, DC & territorial seats
]

JUDICIAL_DESC_TOKENS = [
    # Exact court names or common stems
    "united states district judge",
    "united states circuit judge",
    "court of appeals",                   # covers “United States Court of Appeals…”
    "district court",                     # fallback for shorter phrasing
    "court of international trade",
    "court of federal claims",
    "court of appeals for the federal circuit",
    "court of appeals for veterans",      # “…Veterans Claims” & “…Veterans Appeals”
    "superior court of the district of columbia",
    "district of columbia court of appeals",
    "district court for the northern mariana islands",
    # Generic judicial titles (last, broadest tier)
    "associate justice",
    "chief justice",
    "judge",                              # captures “bankruptcy judge”, etc.
    "justice",
    "magistrate",
]


def is_likely_judicial_summary(nom: dict) -> bool:
    if not isinstance(nom, dict):
        return False

    org  = str(nom.get("organization", "")).lower()
    desc = str(nom.get("description",   "")).lower()

    if any(tok in org for tok in JUDICIAL_ORG_TOKENS):
        return True

    combo = f"{desc} {org}"
    return any(tok in combo for tok in JUDICIAL_DESC_TOKENS)


def is_judicial_nomination(nom: dict) -> bool:
    if not isinstance(nom, dict):
        return False

    # 1️⃣ organization shortcut
    org = str(nom.get("organization", "")).lower()
    if any(tok in org for tok in JUDICIAL_ORG_TOKENS):
        return True

    # 2️⃣ aggregate text fields
    txt = " ".join(
        str(nom.get(k, "")).lower()
        for k in ("description", "positionTitle", "latest_action_text")
    )

    # 3️⃣ nominee-level check (detail responses)
    if "nominees" in nom:
        items = (
            nom["nominees"].get("items", [])
            if isinstance(nom["nominees"], dict)
            else nom["nominees"]
        )
        for n in items:
            txt += " " + str(n.get("positionTitle", "")).lower()
            txt += " " + str(n.get("organization",  "")).lower()

    return any(tok in txt for tok in JUDICIAL_DESC_TOKENS)


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
        for i, nominee in enumerate(nominees_data, 1):
            try:
                logger.debug(f"Processing nominee {i}/{len(nominees_data)}")
                record = base_record.copy()
                
                # Nominee information
                first_name = nominee.get("firstName", "")
                middle_name = nominee.get("middleName", "")
                last_name = nominee.get("lastName", "")
                full_name = " ".join(filter(None, [first_name, middle_name, last_name]))
                
                record["nominee"] = full_name
                record["state"] = nominee.get("state")
                
                # Position information
                position = nominee.get("positionTitle", "")
                organization = nominee.get("organization", "")
                record["position"] = position
                record["organization"] = organization
                
                logger.debug(f"  Name: {full_name}")
                logger.debug(f"  Position: {position} at {organization}")
                
                # Extract circuit/court information from position title
                circuit, court = parse_court_from_description(position)
                if circuit is not None:
                    record["circuit"] = circuit
                    logger.debug(f"  Extracted circuit: {circuit}")
                if court is not None:
                    record["court"] = court
                    logger.debug(f"  Extracted court: {court}")
                
                records.append(record)
                logger.debug(f"  Successfully processed nominee {i}")
                
            except Exception as e:
                logger.error(f"Error processing nominee {i}: {str(e)}")
                logger.debug(f"Problematic nominee data: {nominee}")
                continue
    
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
        default_params = {"api_key": self.api_key, "format": "json", "limit": 500}
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
        try:
            response = requests.get(url_path, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract the nomination from the nested response
            nomination_data = data.get('nomination', {})
            
            # Log basic info about the response
            logger.debug(f"Received detail for nomination {nomination_number}")
            logger.trace(f"Full detail response: {data}")
            
            # Log important fields for debugging
            if 'description' in nomination_data:
                logger.debug(f"Nomination description: {nomination_data['description']}")
            
            # Handle nominees data structure correctly
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