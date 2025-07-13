"""Client for fetching judicial nomination data from Congress.gov API."""

from datetime import datetime, timedelta
import logging
import os
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.auto import tqdm

# Configure standard logger for tenacity (which doesn't work with loguru)
_tenacity_logger = logging.getLogger("tenacity")
_tenacity_logger.setLevel(logging.WARNING)
if not _tenacity_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    _tenacity_logger.addHandler(handler)

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
            "Northern Mariana Islands": "MP", "American Samoa": "AS"
        }
        
        state_code = state_mapping.get(state_name)
        if state_code:
            if direction:
                court = f"{direction}{state_code}"  # e.g., "NCA" for Northern District of California
            else:
                court = state_code  # e.g., "CA" for District of California
    
    return circuit, court


def extract_nomination_data(
    nomination_data: Dict[str, Any], full_details: bool = False
) -> List[Dict[str, Any]]:
    """
    Extract data from a nomination record.
    
    Args:
        nomination_data: Nomination data from API
        full_details: Whether this is detailed nomination data
        
    Returns:
        List of dictionaries with extracted data
    """
    records = []
    
    # Create metadata for each record to track its source
    metadata = {
        "retrieval_date": datetime.now().strftime('%Y-%m-%d'),
        "is_full_detail": full_details,
    }

    # First handle the nomination-level data
    nomination_desc = nomination_data.get("description", "")
    citation = nomination_data.get("citation", "")

    # Parse circuit and court if present in description
    circuit_num, court_code = parse_court_from_description(nomination_desc)
    
    # Add parsed values to metadata
    if circuit_num:
        metadata["parsed_circuit"] = circuit_num
    if court_code:
        metadata["parsed_court"] = court_code
    
    # Get the nominees data
    nominees_data = []
    
    # Handle the nominees collection
    if full_details:
        # For detailed records, nominees are in a structured format
        nominees_data = nomination_data.get("nominees", [])
    else:
        # For summary records, we don't have structured nominee data
        nominees_data = []
    
    # If there are no nominees, create a single record with nomination data
    if not nominees_data:
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
    
    # Properly place the decorator directly above the method
    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
    )            
    def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the Congress.gov API with automatic retries.
        
        Args:
            url: Full API URL
            params: Query parameters to include
            
        Returns:
            API response data
            
        Raises:
            requests.exceptions.RequestException: If the request fails after all retries
        """
        # Add API key to parameters
        params = params or {}
        params["api_key"] = self.api_key
        
        # Track this request for rate limit monitoring
        self._track_request()
        
        # Check if we might be approaching/exceeding rate limit
        status = self.get_rate_limit_status()
        if status["is_exceeded"]:
            logger.error(f"Rate limit exceeded ({status['percent_used']:.1f}%). Request may fail.")
        elif status["is_warning"]:
            logger.warning(f"Approaching rate limit ({status['percent_used']:.1f}%).")
        
        # Make the request
        response = requests.get(url, params=params)
        
        # Handle HTTP errors
        if response.status_code == 429:
            logger.warning(f"Rate limit hit (HTTP 429). Will retry after backoff.")
            response.raise_for_status()  # This will be caught by the retry decorator
            
        if response.status_code != 200:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            response.raise_for_status()
            
        # Parse and return JSON data
        try:
            return response.json()
        except ValueError:
            logger.error(f"Invalid JSON response: {response.text[:100]}...")
            raise ValueError(f"Invalid JSON response from API: {response.text[:100]}...")
    
    # Properly place the decorator directly above the method
    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
    )
    def get_nominations(
        self, congress: int, params: Optional[Dict[str, Any]] = None, auto_paginate: bool = True
    ) -> Dict[str, Any]:
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
        # Check rate limit status before starting
        self.print_rate_limit_summary()
        
        # First get all civilian nominations (exclude military)
        params = {} if params is None else params.copy()
        params["isCivilian"] = "true"
        
        endpoint = f"/nomination/{congress}"
        url = f"{self.BASE_URL}{endpoint}"
        
        # Get first page
        response = self._make_request(url, params)
        all_nominations = response
        
        # Follow pagination if auto_paginate is True
        if auto_paginate and "pagination" in response and "next" in response["pagination"]:
            logger.info(f"Auto-paginating nomination results for Congress {congress}")
            
            # Track nominations across pages
            nominations = response.get("nominations", [])
            
            # Keep fetching until we run out of pages
            next_link = response["pagination"]["next"]
            while next_link:
                # Extract link and make request
                # Check if next_link is absolute or relative URL
                if next_link.startswith('http'):
                    next_url = next_link  # Use as is if it's an absolute URL
                else:
                    next_url = f"{self.BASE_URL}{next_link}"  # Prepend base URL for relative paths
                
                logger.debug(f"Fetching next page: {next_url}")
                response = self._make_request(next_url)
                
                # Add to our results
                if "nominations" in response:
                    nominations.extend(response["nominations"])
                    
                # Check for more pages
                next_link = response.get("pagination", {}).get("next")
            
            # Update the aggregated response with all nominations
            all_nominations["nominations"] = nominations
        
        return all_nominations

    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
    )
    def get_nomination_detail(self, congress: int, nomination_number: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific nomination.
        
        Args:
            congress: Congress number (e.g., 118)
            nomination_number: Nomination number (e.g., "PN123")
            
        Returns:
            Detailed nomination data
            
        Raises:
            requests.exceptions.RequestException: If the request fails after all retries
        """
        endpoint = f"/nomination/{congress}/{nomination_number}"
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            return self._make_request(url)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching nomination detail for {nomination_number}: {str(e)}")
            raise
    
    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
    )
    def get_nominee_data(self, nominee_url: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific nominee.
        
        Args:
            nominee_url: Full URL to the nominee API endpoint
            
        Returns:
            Nominee data as received from the API
            
        Raises:
            requests.exceptions.RequestException: If the request fails after all retries
        """
        try:
            # Check if it's a full URL or just a path
            if nominee_url.startswith('http'):
                url = nominee_url
            else:
                url = f"{self.BASE_URL}{nominee_url}"
                
            # Track rate limit
            self._track_request()
            
            return self._make_request(url)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching nominee data from {nominee_url}: {str(e)}")
            raise
    
    def get_all_nominees_data(self, nominee_urls: List[str], show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch data for multiple nominees in bulk.
        
        Args:
            nominee_urls: List of nominee API URLs
            show_progress: Whether to show progress with tqdm (works in notebooks and terminals)
            
        Returns:
            List of nominee data dictionaries with request metadata
        """
        all_nominees = []
        total = len(nominee_urls)
        
        if total == 0:
            logger.warning("No nominee URLs provided")
            return all_nominees
            
        logger.info(f"Preparing to fetch data for {total} nominees")
        
        # Create progress bar - tqdm.auto automatically selects the appropriate class
        # for the current environment (notebook or terminal)
        progress_iterator = tqdm(nominee_urls, total=total, desc="Fetching nominees") if show_progress else nominee_urls
        
        for url in progress_iterator:
            try:
                # Check if we're approaching rate limit
                rate_status = self.get_rate_limit_status()
                
                if rate_status['is_exceeded']:
                    # Calculate backoff time - starts with 60 seconds and can increase
                    # Hour has 3600 seconds, so we want to wait for approximately
                    # enough time for the oldest requests to drop out of the window
                    backoff_seconds = min(60 * (2 ** (rate_status['percent_used'] / 100 - 1)), 900)  # Cap at 15 minutes
                    
                    logger.warning(f"API rate limit exceeded. Backing off for {backoff_seconds:.1f} seconds")
                    
                    if show_progress:
                        original_desc = progress_iterator.desc
                        for remaining in range(int(backoff_seconds), 0, -1):
                            progress_iterator.set_description(
                                f"Rate limit hit - waiting {remaining}s"
                            )
                            time.sleep(1)
                        progress_iterator.set_description(original_desc)
                    else:
                        time.sleep(backoff_seconds)
                        
                    logger.info("Resuming nominee data collection after backoff")
                    
                elif rate_status['is_warning']:
                    warning_msg = f"API rate limit warning: {rate_status['percent_used']:.1f}% used"
                    logger.warning(warning_msg)
                    if show_progress:
                        # Update progress bar description to show warning
                        progress_iterator.set_description(f"Fetching nominees (Rate limit: {rate_status['percent_used']:.1f}%)")
                        
                    # Introduce a small delay when approaching limits
                    if rate_status['percent_used'] > 90:
                        delay = 2.0  # 2 seconds when very close to limit
                    else:
                        delay = 0.5  # 0.5 seconds when just at warning threshold
                        
                    time.sleep(delay)
                
                # Get nominee data
                nominee_data = self.get_nominee_data(url)
                
                # Add request metadata
                request_info = {
                    'request': {'url': url},
                    'retrieval_date': datetime.now().strftime('%Y-%m-%d'),
                }
                
                # Store as a dict with the nominee data and request info
                nominee_entry = {
                    'nominee': nominee_data,
                    'request': request_info['request'],
                    'retrieval_date': request_info['retrieval_date'],
                }
                
                all_nominees.append(nominee_entry)
                    
            except Exception as e:
                error_msg = f"Error processing nominee URL {url}: {str(e)}"
                logger.error(error_msg)
                if show_progress:
                    # Update progress bar postfix to show last error
                    progress_iterator.set_postfix_str(f"Last error: {str(e)[:30]}..." if len(str(e)) > 30 else f"Last error: {str(e)}")
                # Continue with next nominee instead of failing the entire batch
        
        # Print final rate limit status after processing all nominees
        self.print_rate_limit_summary()
        
        logger.info(f"Successfully retrieved data for {len(all_nominees)}/{total} nominees")
            
        return all_nominees
        
    def get_judicial_nominations(self, congress: int, auto_paginate: bool = True) -> List[Dict[str, Any]]:
        """
        Get all judicial nominations for a specific Congress term.
        
        Args:
            congress: Congress number (e.g., 118)
            auto_paginate: Whether to automatically paginate API results
            
        Returns:
            List of processed nomination records
            
        Raises:
            RuntimeError: If the API rate limit is exceeded
        """
        try:
            # First get all civilian nominations
            all_nominations = self.get_nominations(congress, auto_paginate=auto_paginate)
            
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
        
        # Create tqdm progress bar if we have more than a few nominations
        use_progress_bar = len(judicial_nominations) > 10
        nominations_iter = tqdm(judicial_nominations, desc="Fetching nomination details") if use_progress_bar else judicial_nominations
        
        for i, nomination in enumerate(nominations_iter):
            congress_number = nomination.get("congress")
            nomination_number = nomination.get("number")
            
            if not congress_number or not nomination_number:
                logger.warning(f"Missing congress or nomination number for entry {i}")
                continue
                
            if use_progress_bar:
                nominations_iter.set_description(f"Processing: {nomination_number}")
            else:
                logger.info(f"Processing judicial nomination {i+1}/{len(judicial_nominations)}: {nomination_number}")
            
            # Check rate limits and apply graduated delays if needed
            rate_status = self.get_rate_limit_status()
            
            if rate_status['is_exceeded']:
                # Calculate backoff time - starts with 60 seconds and can increase
                # Cap at 15 minutes to avoid excessive waits
                backoff_seconds = min(60 * (2 ** (rate_status['percent_used'] / 100 - 1)), 900)
                
                logger.warning(f"API rate limit exceeded. Backing off for {backoff_seconds:.1f} seconds")
                
                if use_progress_bar:
                    original_desc = nominations_iter.desc
                    for remaining in range(int(backoff_seconds), 0, -1):
                        nominations_iter.set_description(f"Rate limit hit - waiting {remaining}s")
                        time.sleep(1)
                    nominations_iter.set_description(original_desc)
                else:
                    time.sleep(backoff_seconds)
                    
                logger.info("Resuming nomination data collection after backoff")
                
            elif rate_status['is_warning']:
                warning_msg = f"API rate limit warning: {rate_status['percent_used']:.1f}% used"
                logger.warning(warning_msg)
                
                if use_progress_bar:
                    # Update progress bar description to show warning
                    nominations_iter.set_description(f"Processing: {nomination_number} (Rate: {rate_status['percent_used']:.1f}%)")
                
                # Introduce a small delay when approaching limits
                if rate_status['percent_used'] > 90:
                    delay = 2.0  # 2 seconds when very close to limit
                else:
                    delay = 0.5  # 0.5 seconds when just at warning threshold
                    
                time.sleep(delay)
            
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
