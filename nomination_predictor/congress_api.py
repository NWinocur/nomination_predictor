"""Client for fetching judicial nomination data from Congress.gov API."""

from datetime import datetime, timedelta
import os
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger
import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Tokens that may indicate a judicial nomination when found in description/position/text etc. fields
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


# normalize_column_names function has been removed
# Column name normalization is now handled in the data cleaning notebook
# to maintain separation of concerns between data fetching and data processing


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
        """Initialize the Congress.gov API client.
        
        Args:
            api_key: API key for Congress.gov. If not provided,
                attempts to use CONGRESS_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("CONGRESS_API_KEY")
        
        # Rate limit tracking properties
        self.request_timestamps: List[datetime] = []
        self.max_hourly_requests: int = 5000
        self.warning_threshold: float = 0.8  # 80% of limit
        
        # Retry configuration
        self.max_retries = 5
        self.min_wait_seconds = 2
        self.max_wait_seconds = 60
        
        if not self.api_key:
            raise ValueError("Congress.gov API key is required")
            
    def create_retry_decorator(self) -> Callable:
        """
        DEPRECATED: This method is no longer used. Retry decorators are now applied
        directly to methods for clarity and maintainability.
        
        All API methods now use the same retry configuration:
        - Retries on network errors (requests.exceptions.RequestException, ConnectionError)
        - Uses exponential backoff starting at 2s, doubling each retry up to 60s max wait
        - Gives up after 5 attempts
        - Logs warnings before each retry
        
        The retry configuration provides resilience against:
        - Transient network issues
        - Rate limiting (HTTP 429 responses)
        - Brief API outages
        
        NOTE: This retry mechanism handles retrying individual API calls. It does not
        implement partial-results handling or resume-on-restart for bulk operations,
        which would require additional state management.
        """
        # This method is kept for backward compatibility but is no longer used
        return retry(
            # Retry conditions - retry on requests exceptions and HTTP 429 (rate limit)
            retry=retry_if_exception_type((
                requests.exceptions.RequestException, 
                requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout
            )),
            # Stop after max_retries attempts
            stop=stop_after_attempt(self.max_retries),
            # Wait exponentially between retries, starting at min_wait_seconds and 
            # increasing up to max_wait_seconds with randomized jitter
            wait=wait_exponential(multiplier=1, min=self.min_wait_seconds, max=self.max_wait_seconds),
            # Log before each retry with detailed information
            before_sleep=before_sleep_log(logger, level='WARNING'),
            # Retry even if we get a response with status code 429
            reraise=True
        )
            
    def _track_request(self, count_request: bool = True) -> None:
        """Track a new API request and clean up old request timestamps.
        
        Args:
            count_request: Whether to add a new timestamp for the current request
            
        Returns:
            None
        """
        now = datetime.now()
        
        # Add current request timestamp if this is an actual request
        if count_request:
            self.request_timestamps.append(now)
        
        # Remove timestamps older than 1 hour
        one_hour_ago = now - timedelta(hours=1)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts >= one_hour_ago]
        
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current API rate limit status.
        
        Returns:
            Dict with rate limit information:
                - requests_made: Number of requests in the last hour
                - requests_remaining: Number of requests remaining in the current hour
                - percent_used: Percentage of hourly limit used
                - is_warning: Whether usage is above warning threshold
                - is_exceeded: Whether rate limit is exceeded
        """
        # Clean up old timestamps first without counting this status check as a request
        self._track_request(count_request=False)
        
        requests_made = len(self.request_timestamps)
        requests_remaining = max(0, self.max_hourly_requests - requests_made)
        percent_used = (requests_made / self.max_hourly_requests) * 100 if self.max_hourly_requests > 0 else 100
        
        return {
            "requests_made": requests_made,
            "requests_remaining": requests_remaining, 
            "percent_used": percent_used,
            "is_warning": percent_used >= (self.warning_threshold * 100),
            "is_exceeded": requests_made >= self.max_hourly_requests
        }
        
    def print_rate_limit_summary(self) -> None:
        """Print a summary of the current rate limit status.
        
        This is useful for callers to check after bulk operations.
        """
        status = self.get_rate_limit_status()
        
        if status["is_exceeded"]:
            logger.error(
                f"⚠️ RATE LIMIT EXCEEDED: {status['requests_made']}/{self.max_hourly_requests} "
                f"requests ({status['percent_used']:.1f}%) in the last hour"
            )
        elif status["is_warning"]:
            logger.warning(
                f"⚠️ APPROACHING RATE LIMIT: {status['requests_made']}/{self.max_hourly_requests} "
                f"requests ({status['percent_used']:.1f}%) in the last hour"
            )
        else:
            logger.info(
                f"✓ Rate limit status: {status['requests_made']}/{self.max_hourly_requests} "
                f"requests ({status['percent_used']:.1f}%) in the last hour"
            )
            
    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),  # Start at 2s, double each retry, max 60s
        stop=stop_after_attempt(5),  # Give up after 5 attempts
        before_sleep=before_sleep_log(logger, level="WARNING"),
    )
    
    def get_nominations(
        self, congress: int, params: Dict[str, Any] = None, auto_paginate:bool=True
    ) -> List[Dict[str, Any]]:
        """
        Fetch nominations for a specific congress with pagination support.
        
        This method automatically handles pagination by following the 'next' link
        in the API response until all records are retrieved.
        
        Args:
            congress: Congress number (e.g., 118)
            params: Additional parameters to pass to the API
            auto_paginate: Whether to automatically fetch all pages (default: True)
            
        Returns:
            Combined results from all pages with 'nominations' key containing
            all nominations across all pages
            
        Raises:
            RuntimeError: If the API rate limit is exceeded
        """
        # Print initial rate limit status
        self.print_rate_limit_summary()
        
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
        
        # Print rate limit status before detailed fetching
        self.print_rate_limit_summary()
        
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
            except requests.exceptions.RequestException as e:
                # This exception would have been retried by the decorator on get_nomination_detail
                # If we're here, we've exhausted all retries
                logger.error(f"Error processing nomination {nomination_number} after all retries: {str(e)}")
                # Fall back to summary records after retry attempts
                detailed_records.extend(summary_records)
                logger.info(f"Added {len(summary_records)} summary-level records for nomination {nomination_number} (after retry failure)")
                # Print traceback for easier debugging
                traceback.print_exc()
        
        # Print final rate limit status after processing all nominations
        self.print_rate_limit_summary()
                
        logger.info(f"Found {len(detailed_records)} detailed judicial nomination records")
        
        # No longer normalizing column names here - this will be done in the data cleaning notebook
        logger.info(f"Processed {len(detailed_records)} total judicial nomination records for Congress {congress}")
        return detailed_records

    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),  # Start at 2s, double each retry, max 60s
        stop=stop_after_attempt(5),  # Give up after 5 attempts
        before_sleep=before_sleep_log(logger, level="WARNING"),
    )
    def get_nomination_detail(self, congress: int, nomination_number: int, part_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch detailed information about a specific nomination.
        
        Args:
            congress: Congress number (e.g., 118)
            nomination_number: Nomination number (e.g., 1)
            part_number: Optional part number for multi-part nominations
            
        Returns:
            Nomination detail data from the API
        """
        # Construct URL based on parameters
        url = f"{self.BASE_URL}/nomination/{congress}/{nomination_number}"
        
        # Add part number if provided
        if part_number:
            url = f"{url}/{part_number}"
        
        params = {
            "api_key": self.api_key,
            "format": "json",
        }
        
        logger.info(f"Fetching nomination detail from {url}")
        self._track_request()
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching nomination detail: {str(e)}")
            raise  # Will be retried by the decorator

    def get_judicial_nominations(self, congress: int, auto_paginate:bool=True) -> List[Dict[str, Any]]:
        """
        Fetch and filter for judicial nominations from a single congress.
        
        Uses only summary-level filtering to identify judicial nominations,
        then retrieves detailed data for each one that passes the filter.
        Returns raw API data with minimal processing (column normalization only).

        Args:
            congress: Congress number (e.g., 118)
            auto_paginate: Whether to automatically fetch all pages (default: True)

        Returns:
            List of judicial nomination records with minimal processing
            
        Raises:
            RuntimeError: If the API rate limit is exceeded
        """
        logger.info(f"Fetching judicial nominations for Congress {congress}")
        
        # Print initial rate limit status
        self.print_rate_limit_summary()
        
        try:
            # First get all civilian nominations (exclude military)
            # Note: get_nominations already has its own retry logic
            all_nominations = self.get_nominations(congress, {"isCivilian": "true"}, auto_paginate=auto_paginate)    
            
            if not all_nominations or "nominations" not in all_nominations:
                logger.warning(f"No nominations found for Congress {congress}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching nominations for Congress {congress} after retries: {str(e)}")
            raise RuntimeError(f"Failed to fetch nominations for Congress {congress} after multiple retries: {str(e)}")
        
        # Get the actual nomination items from the response
        nominations = all_nominations.get("nominations", [])
        logger.info(f"Found {len(nominations)} civilian nominations in Congress {congress}")
        
        # Filter for likely judicial nominations using summary data
        judicial_nominations = [n for n in nominations if is_likely_judicial_summary(n)]
        logger.info(f"Found {len(judicial_nominations)} judicial nominations based on summary data")
        
        # Log some examples of what we found
        for i, nom in enumerate(judicial_nominations[:10]):
            logger.info(f"Judicial nomination {i+1}: {nom.get('number')} - {nom.get('description', 'No description')}")
        
        # Print rate limit status before detailed fetching
        self.print_rate_limit_summary()
        
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
            except requests.exceptions.RequestException as e:
                # This exception would have been retried by the decorator on get_nomination_detail
                # If we're here, we've exhausted all retries
                logger.error(f"Error processing nomination {nomination_number} after all retries: {str(e)}")
                # Fall back to summary records after retry attempts
                detailed_records.extend(summary_records)
                logger.info(f"Added {len(summary_records)} summary-level records for nomination {nomination_number} (after retry failure)")
                # Print traceback for easier debugging
                traceback.print_exc()
        
        # Print final rate limit status after processing all nominations
        self.print_rate_limit_summary()
                
        logger.info(f"Found {len(detailed_records)} detailed judicial nomination records")
        
        logger.info(f"Processed {len(detailed_records)} total judicial nomination records for Congress {congress}")
        return detailed_records
