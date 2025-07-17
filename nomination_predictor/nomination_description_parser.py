"""
Nomination Description Parser

A unified parser for Congressional nomination descriptions, inspired by nameparser's HumanName.
This module provides a comprehensive solution for parsing nomination descriptions into structured components.

The typical format is:
"Full name, of the US_state_or_location, to be a/an position_title for/of the court_name, 
(for a/an optional term limit), vice predecessor_name, reason."

"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List

import pandas as pd


@dataclass
class NominationDescription:
    """
    Structured representation of a nomination description, similar to nameparser.HumanName.
    
    This class parses nomination descriptions and provides easy access to all components.
    """
    
    # Core components
    nominee_name: str = ""
    location: str = ""
    position_title: str = ""
    court_name: str = ""
    term_info: str = ""
    predecessor_name: str = ""
    vacancy_reason: str = ""
    
    # Raw and processed data
    raw_description: str = ""
    is_parsed: bool = False
    parsing_confidence: str = "unknown"  # high, medium, low, failed
    
    # Additional metadata
    is_list_nomination: bool = False
    multiple_nominees: List[str] = None
    
    def __post_init__(self):
        """Initialize the multiple_nominees list if None."""
        if self.multiple_nominees is None:
            self.multiple_nominees = []
    
    @classmethod
    def from_description(cls, description: str) -> 'NominationDescription':
        """
        Create a NominationDescription from a raw description string.
        
        Args:
            description: Raw nomination description text
            
        Returns:
            NominationDescription instance with parsed components
        """
        if pd.isna(description) or not description.strip():
            return cls(raw_description=str(description), parsing_confidence="failed")
        
        instance = cls(raw_description=description)
        instance._parse()
        return instance
    
    def _parse(self) -> None:
        """Parse the raw description into structured components."""
        if not self.raw_description or pd.isna(self.raw_description):
            self.parsing_confidence = "failed"
            return
        
        description = self.raw_description.strip()
        
        # Check for list nominations (multiple nominees)
        if self._is_list_nomination(description):
            self._parse_list_nomination(description)
            return
        
        # Parse single nomination
        self._parse_single_nomination(description)
    
    def _is_list_nomination(self, description: str) -> bool:
        """Check if this is a list nomination with multiple nominees."""
        list_indicators = [
            "the following-named persons",
            "the following named persons", 
            "\\t",  # Tab-separated lists
            "\\n",  # Newline-separated lists
        ]
        
        description_lower = description.lower()
        for indicator in list_indicators:
            if indicator in description_lower:
                self.is_list_nomination = True
                return True
        
        return False
    
    def _parse_list_nomination(self, description: str) -> None:
        """Parse a list nomination with multiple nominees."""
        self.parsing_confidence = "medium"
        self.is_parsed = True
        
        # Extract the general position and court info from the beginning
        position_match = re.search(r'to be (.+?)(?:\.|$)', description, re.IGNORECASE)
        if position_match:
            position_text = position_match.group(1).strip()
            
            # Split position and court info
            court_patterns = [
                r'(?:for|of) the (.+?)(?:\s+for\s+|$)',
                r'(?:for|of) (.+?)(?:\s+for\s+|$)',
            ]
            
            for pattern in court_patterns:
                court_match = re.search(pattern, position_text, re.IGNORECASE)
                if court_match:
                    self.court_name = court_match.group(1).strip()
                    self.position_title = position_text.replace(court_match.group(0), '').strip()
                    break
            else:
                self.position_title = position_text
        
        # Extract term information
        term_match = re.search(r'for (?:a )?term(?:s)? (?:of )?(.+?)(?:\.|,|$)', description, re.IGNORECASE)
        if term_match:
            self.term_info = term_match.group(1).strip()
        
        # Extract individual nominees (simplified for list nominations)
        # This is a basic implementation - could be enhanced for specific list formats
        tab_separated = re.findall(r'\\t([^\\t]+)', description)
        if tab_separated:
            self.multiple_nominees = [name.strip() for name in tab_separated if name.strip()]
        
        # If no tab separation, try to extract names from the description
        if not self.multiple_nominees:
            # Look for patterns like "Name, of State"
            name_patterns = re.findall(r'([A-Z][a-z]+ [A-Z][a-z]*(?:\s+[A-Z][a-z]*)*),\s+of\s+([^,]+)', description)
            if name_patterns:
                self.multiple_nominees = [f"{name}, of {location}" for name, location in name_patterns]
    
    def _parse_single_nomination(self, description: str) -> None:
        """Parse a single nomination description."""
        
        # Main parsing patterns for single nominations
        patterns = [
            # Standard pattern: "Name, of Location, to be Position for/of Court, [term info], vice Predecessor, reason"
            r'^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?)(?:\s+for\s+(?:a\s+)?term\s+(?:of\s+)?(.+?))?(?:,\s+vice\s+(.+?),\s+(.+?))?\.?$',
            
            # Pattern without predecessor: "Name, of Location, to be Position for/of Court, [term info]"
            r'^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?)(?:\s+for\s+(?:a\s+)?term\s+(?:of\s+)?(.+?))?\.?$',
            
            # Pattern with court in position: "Name, of Location, to be Position"
            r'^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?)\.?$',
        ]
        
        parsed = False
        
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE | re.DOTALL)
            if match:
                self._extract_components_from_match(match, description)
                parsed = True
                break
        
        if not parsed:
            # Fallback parsing for non-standard formats
            self._fallback_parse(description)
        
        self.is_parsed = True
    
    def _extract_components_from_match(self, match: re.Match, description: str) -> None:
        """Extract components from a regex match."""
        groups = match.groups()
        
        # Basic components
        self.nominee_name = groups[0].strip() if groups[0] else ""
        self.location = groups[1].strip() if len(groups) > 1 and groups[1] else ""
        
        # Position and court extraction
        if len(groups) > 2 and groups[2]:
            position_text = groups[2].strip()
            self._extract_position_and_court(position_text)
        
        # Term information
        if len(groups) > 3 and groups[3]:
            self.term_info = groups[3].strip()
        
        # Predecessor and reason (from vice clause)
        if len(groups) > 4 and groups[4]:
            self.predecessor_name = groups[4].strip()
        if len(groups) > 5 and groups[5]:
            self.vacancy_reason = groups[5].strip()
        
        # If no predecessor found in main pattern, try to extract vice clause
        if not self.predecessor_name:
            self._extract_vice_clause(description)
        
        self.parsing_confidence = "high"
    
    def _extract_position_and_court(self, position_text: str) -> None:
        """Extract position title and court name from position text."""
        
        # Patterns to separate position from court
        court_patterns = [
            # "Position for the Court Name"
            r'^(.+?)\s+for\s+the\s+(.+)$',
            # "Position of the Court Name"  
            r'^(.+?)\s+of\s+the\s+(.+)$',
            # "Position for Court Name"
            r'^(.+?)\s+for\s+(.+)$',
            # "Position of Court Name"
            r'^(.+?)\s+of\s+(.+)$',
        ]
        
        for pattern in court_patterns:
            match = re.search(pattern, position_text, re.IGNORECASE)
            if match:
                self.position_title = match.group(1).strip()
                self.court_name = match.group(2).strip()
                return
        
        # If no court pattern matches, the entire text is the position
        self.position_title = position_text
    
    def _extract_vice_clause(self, description: str) -> None:
        """Extract predecessor and vacancy reason from vice clause."""
        
        # Patterns for vice clauses
        vice_patterns = [
            # "vice Name, reason"
            r'vice\s+(.+?),\s+(.+?)(?:\.|$)',
            # "vice Name reason" (no comma)
            r'vice\s+(.+?)\s+(retired|deceased|resigned|elevated)(?:\.|$)',
            # Just "vice Name"
            r'vice\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in vice_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                self.predecessor_name = match.group(1).strip()
                if len(match.groups()) > 1:
                    self.vacancy_reason = match.group(2).strip()
                break
    
    def _fallback_parse(self, description: str) -> None:
        """Fallback parsing for non-standard formats."""
        self.parsing_confidence = "low"
        
        # Try to extract name (first part before comma)
        name_match = re.search(r'^([^,]+)', description)
        if name_match:
            self.nominee_name = name_match.group(1).strip()
        
        # Try to extract location
        location_match = re.search(r'of\s+(?:the\s+)?([^,]+)', description, re.IGNORECASE)
        if location_match:
            self.location = location_match.group(1).strip()
        
        # Try to extract position
        position_match = re.search(r'to\s+be\s+(.+?)(?:\s+for\s+|\s+of\s+|,|\.)', description, re.IGNORECASE)
        if position_match:
            position_text = position_match.group(1).strip()
            self._extract_position_and_court(position_text)
        
        # Try to extract vice clause
        self._extract_vice_clause(description)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy DataFrame integration."""
        return {
            'nominee_name': self.nominee_name,
            'nomination_of_or_from_location': self.location,
            'nomination_to_position_title': self.position_title,
            'nomination_to_court_name': self.court_name,
            'nomination_to_term_info': self.term_info,
            'nomination_predecessor_name': self.predecessor_name,
            'nomination_vacancy_reason': self.vacancy_reason,
            'nomination_is_list_nomination': self.is_list_nomination,
            'nomination_parsing_confidence': self.parsing_confidence,
            'nomination_multiple_nominees_count': len(self.multiple_nominees),
        }
    
    def __str__(self) -> str:
        """String representation showing key components."""
        if self.is_list_nomination:
            return f"List Nomination: {len(self.multiple_nominees)} nominees for {self.position_title}"
        else:
            return f"{self.nominee_name} → {self.position_title} ({self.court_name})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"NominationDescription(name='{self.nominee_name}', court='{self.court_name}', confidence='{self.parsing_confidence}')"


def parse_nomination_descriptions(descriptions: pd.Series) -> pd.DataFrame:
    """
    Parse a pandas Series of nomination descriptions into a DataFrame of components.
    
    Args:
        descriptions: Series containing nomination description strings
        
    Returns:
        DataFrame with parsed components as columns
    """
    parsed_data = {}
    
    for idx, desc in descriptions.items():
        nomination = NominationDescription.from_description(desc)
        parsed_data[idx] = nomination.to_dict()
    
    return pd.DataFrame.from_dict(parsed_data, orient='index')


def extract_nominee_name(description: str) -> str:
    """Extract nominee name from description (backward compatibility)."""
    nomination = NominationDescription.from_description(description)
    return nomination.nominee_name


def extract_location(description: str) -> str:
    """Extract location from description (backward compatibility)."""
    nomination = NominationDescription.from_description(description)
    return nomination.location


def extract_court_name(description: str) -> str:
    """Extract court name from description (backward compatibility)."""
    nomination = NominationDescription.from_description(description)
    return nomination.court_name


def extract_position_title(description: str) -> str:
    """Extract position title from description (backward compatibility)."""
    nomination = NominationDescription.from_description(description)
    return nomination.position_title


def extract_predecessor_name(description: str) -> str:
    """Extract predecessor name from description (backward compatibility)."""
    nomination = NominationDescription.from_description(description)
    return nomination.predecessor_name


def extract_vacancy_reason(description: str) -> str:
    """Extract vacancy reason from description (backward compatibility)."""
    nomination = NominationDescription.from_description(description)
    return nomination.vacancy_reason


# Convenience function for batch processing
def parse_descriptions_to_columns(df: pd.DataFrame, add_prefix: bool = False) -> pd.DataFrame:
    """
    Parse nomination descriptions and add parsed components as new columns to DataFrame.
    
    Args:
        df: DataFrame containing nomination descriptions
        add_prefix: If True, add 'parsed_' prefix to column names. If False, use direct column names.
        
    Returns:
        DataFrame with additional parsed component columns
    """
    if "description" not in df.columns:
        raise ValueError("Column `description` not found in DataFrame")
    
    # Parse descriptions
    parsed_df = parse_nomination_descriptions(df["description"])
    
    # Add parsed columns to DataFrame
    result_df = df.copy()
    for col in parsed_df.columns:
        if add_prefix:
            result_df[f'parsed_{col}'] = parsed_df[col]
        else:
            result_df[col] = parsed_df[col]
    
    return result_df
