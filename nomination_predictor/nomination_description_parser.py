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
from typing import Any, Dict, List, Optional

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
    predecessor_name: Optional[str] = None
    vacancy_reason: Optional[str] = None

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
    def from_description(cls, description: str) -> "NominationDescription":
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
        position_match = re.search(r"to be (.+?)(?:\.|$)", description, re.IGNORECASE)
        if position_match:
            position_text = position_match.group(1).strip()

            # Split position and court info
            court_patterns = [
                r"(?:for|of) the (.+?)(?:\s+for\s+|$)",
                r"(?:for|of) (.+?)(?:\s+for\s+|$)",
            ]

            for pattern in court_patterns:
                court_match = re.search(pattern, position_text, re.IGNORECASE)
                if court_match:
                    self.court_name = court_match.group(1).strip()
                    self.position_title = position_text.replace(court_match.group(0), "").strip()
                    break
            else:
                self.position_title = position_text

        # Extract term information
        term_match = re.search(
            r"for (?:a )?term(?:s)? (?:of )?(.+?)(?:\.|,|$)", description, re.IGNORECASE
        )
        if term_match:
            self.term_info = term_match.group(1).strip()

        # Extract individual nominees (simplified for list nominations)
        # This is a basic implementation - could be enhanced for specific list formats
        tab_separated = re.findall(r"\\t([^\\t]+)", description)
        if tab_separated:
            self.multiple_nominees = [name.strip() for name in tab_separated if name.strip()]

        # If no tab separation, try to extract names from the description
        if not self.multiple_nominees:
            # Look for patterns like "Name, of State"
            name_patterns = re.findall(
                r"([A-Z][a-z]+ [A-Z][a-z]*(?:\s+[A-Z][a-z]*)*),\s+of\s+([^,]+)", description
            )
            if name_patterns:
                self.multiple_nominees = [
                    f"{name}, of {location}" for name, location in name_patterns
                ]

    def _parse_single_nomination(self, description: str) -> None:
        """Parse a single nomination description."""

        # Main parsing patterns for single nominations
        patterns = [
            # Pattern with term and vice clause: "Name, of Location, to be Position for term, vice Predecessor, reason"
            r"^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?)\s+for\s+(?:a\s+)?term\s+(?:of\s+)?(.+?),\s+vice\s+(.+?),\s+(.+?)\.?$",
            # Pattern with term but no vice clause: "Name, of Location, to be Position for term"
            r"^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?)\s+for\s+(?:a\s+)?term\s+(?:of\s+)?(.+?)\.?$",
            # Pattern with vice clause but no term: "Name, of Location, to be Position, vice Predecessor, reason"
            r"^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?),\s+vice\s+(.+?),\s+(.+?)\.?$",
            # Pattern with vice clause but no explicit reason: "Name, of Location, to be Position vice Predecessor, reason"
            r"^(.+?),\s+of\s+(.+?)\s+to\s+be\s+(.+?)\s+vice\s+(.+?),\s+(.+?)\.?$",
            # Pattern with vice clause but no comma before vice: "Name, of Location, to be Position of Court vice Predecessor reason"
            r"^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?)\s+vice\s+(.+?)\s+(retired|deceased|resigned|elevated)\.?$",
            # Pattern with missing space after location comma: "Name, of Location,to be Position..."
            r"^(.+?),\s+of\s+(.+?),?to\s+be\s+(.+?)\s+vice\s+(.+?),\s+(.+?)\.?$",
            # Pattern with "to the" instead of "to be": "Name, of Location, to the Position..."
            r"^(.+?),\s+of\s+(.+?),\s+to\s+the\s+(.+?)\s+vice\s+(.+?),\s+(.+?)\.?$",
            # Simple pattern without predecessor: "Name, of Location, to be Position of Court"
            r"^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?)\.?$",
            # Simple pattern with "to the": "Name, of Location, to the Position of Court"
            r"^(.+?),\s+of\s+(.+?),\s+to\s+the\s+(.+?)\.?$",
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
        location = groups[1].strip() if len(groups) > 1 and groups[1] else ""

        # Clean up location - remove "the" prefix and trailing punctuation
        location = re.sub(r"^the\s+", "", location, flags=re.IGNORECASE)
        location = re.sub(r"[,;.]+$", "", location)  # Remove trailing punctuation
        self.location = location

        # Position and court extraction
        if len(groups) > 2 and groups[2]:
            position_text = groups[2].strip()
            self._extract_position_and_court(position_text)

        # Handle different pattern structures based on actual regex patterns matched
        # Pattern 1: Name, Location, Position+Court, Term, Predecessor, Reason (6 groups)
        if len(groups) == 6 and "term" in description.lower() and "vice" in description.lower():
            if groups[3]:  # term info
                self.term_info = groups[3].strip()
            if groups[4]:  # predecessor
                cleaned_predecessor = self._clean_predecessor_name(groups[4].strip())
                self.predecessor_name = cleaned_predecessor if cleaned_predecessor else None
            if groups[5]:  # vacancy reason
                cleaned_reason = self._clean_vacancy_reason(groups[5].strip())
                self.vacancy_reason = cleaned_reason if cleaned_reason else None
        # Pattern 2: Name, Location, Position+Court, Term (4 groups)
        elif len(groups) == 4 and "term" in description.lower():
            if groups[3]:  # term info
                self.term_info = groups[3].strip()
        # Pattern 3: Name, Location, Position+Court, Predecessor, Reason (5 groups)
        elif len(groups) == 5 and "vice" in description.lower():
            if groups[3]:  # predecessor
                cleaned_predecessor = self._clean_predecessor_name(groups[3].strip())
                self.predecessor_name = cleaned_predecessor if cleaned_predecessor else None
            if groups[4]:  # vacancy reason
                cleaned_reason = self._clean_vacancy_reason(groups[4].strip())
                self.vacancy_reason = cleaned_reason if cleaned_reason else None
        # Pattern 4: Name, Location, Position+Court (3 groups)
        elif len(groups) == 3:
            # No additional info to extract
            pass
        else:
            # Fallback: try to extract vice clause separately
            self._extract_vice_clause(description)

        # If no predecessor found in main pattern, try to extract vice clause
        if not self.predecessor_name:
            self._extract_vice_clause(description)

        self.parsing_confidence = "high"

    def _clean_predecessor_name(self, name: str) -> str:
        """Clean up predecessor name, ensuring suffixes are preserved."""
        if not name:
            return ""

        name = name.strip()

        # Remove trailing periods unless they're part of a suffix
        if re.search(r"(?:Jr|Sr|III|II|IV)\.$", name, re.IGNORECASE):
            # Name ends with a suffix and period - keep as is
            pass
        elif re.search(r"(?:Jr|Sr|III|II|IV)$", name, re.IGNORECASE):
            # Name ends with suffix but no period - add period
            name = re.sub(r"(Jr|Sr|III|II|IV)$", r"\1.", name, flags=re.IGNORECASE)
        else:
            # Remove trailing periods from names without suffixes
            name = re.sub(r"\.$", "", name)

        return name

    def _clean_vacancy_reason(self, reason: str) -> str:
        """Clean up vacancy reason, removing any name suffixes that leaked in."""
        if not reason:
            return ""

        # Only remove suffixes that are clearly part of a name (preceded by comma or at start)
        # Pattern: "Jr., retired" -> "retired"
        reason = re.sub(r"^(?:Jr|Sr|III|II|IV)\.?,\s*", "", reason, flags=re.IGNORECASE)
        # Pattern: "Name Jr., retired" -> "retired" (remove everything up to and including suffix)
        reason = re.sub(r"^.*?(?:Jr|Sr|III|II|IV)\.?,\s*", "", reason, flags=re.IGNORECASE)

        cleaned = reason.strip()
        return cleaned if cleaned else ""

    def _extract_position_and_court(self, position_text: str) -> None:
        """Extract position title and court name from position text."""

        # Remove any vice clause from the position text first
        position_text = re.sub(r"\s+vice\s+.+$", "", position_text, flags=re.IGNORECASE)

        # Patterns to separate position from court - more specific patterns first
        court_patterns = [
            # "an Associate Judge of the Court Name" - handles articles with "the"
            r"^(?:an?\s+)?(Associate Judge)\s+of\s+the\s+(.+)$",
            # "an Associate Judge of Court Name" - handles articles without "the"
            r"^(?:an?\s+)?(Associate Judge)\s+of\s+(.+)$",
            # "an Associate Justice of the Court Name" - handles articles with "the"
            r"^(?:an?\s+)?(Associate Justice)\s+of\s+the\s+(.+)$",
            # "an Associate Justice of Court Name" - handles articles without "the"
            r"^(?:an?\s+)?(Associate Justice)\s+of\s+(.+)$",
            # "Position for the Court Name" - handles "for the" construction
            r"^(.+?)\s+for\s+the\s+(.+)$",
            # "Position of the Court Name" - handles "of the" construction
            r"^(.+?)\s+of\s+the\s+(.+)$",
            # "Position for Court Name" - handles "for" construction
            r"^(.+?)\s+for\s+(.+)$",
            # "Position of Court Name" - handles "of" construction but be more careful
            # This pattern should not match if "of" is part of the position title itself
            r"^((?:United States |Chief |Senior |Magistrate |Circuit |District )?(?:Judge|Attorney|Justice|Commissioner))\s+of\s+(.+)$",
        ]

        for pattern in court_patterns:
            match = re.search(pattern, position_text, re.IGNORECASE)
            if match:
                position_title = match.group(1).strip()
                court_name = match.group(2).strip()

                # Clean up position title - remove articles "a" or "an"
                position_title = re.sub(r"^(?:a|an)\s+", "", position_title, flags=re.IGNORECASE)

                # Clean up court name - remove term information and "the" prefix
                court_name = re.sub(r"^the\s+", "", court_name, flags=re.IGNORECASE)
                # Remove term information more aggressively
                court_name = re.sub(
                    r"\s+for\s+(?:a\s+)?term\s+(?:of\s+)?.*$", "", court_name, flags=re.IGNORECASE
                )
                court_name = re.sub(
                    r"\s+for\s+the\s+term\s+(?:of\s+)?.*$", "", court_name, flags=re.IGNORECASE
                )
                # Remove any trailing punctuation or whitespace
                court_name = re.sub(r"[,\]\s]*$", "", court_name)
                court_name = court_name.strip()

                self.position_title = position_title
                self.court_name = court_name
                return

        # If no court pattern matches, the entire text is the position
        position_title = position_text.strip()
        # Clean up position title - remove articles "a" or "an"
        position_title = re.sub(r"^(?:a|an)\s+", "", position_title, flags=re.IGNORECASE)
        self.position_title = position_title

    def _extract_vice_clause(self, description: str) -> None:
        """Extract predecessor and vacancy reason from vice clause."""

        # Patterns for vice clauses - updated to handle all name formats properly
        vice_patterns = [
            # "vice First Last, Jr., reason" - handles names with comma-separated suffixes (MOST SPECIFIC FIRST)
            r"vice\s+([A-Z][a-zA-Z]*(?:\s+[A-Z]\.?)*\s+[A-Z][a-zA-Z]+,\s+(?:Jr|Sr|III|II|IV)\.?),\s+(.+?)(?:\.|$)",
            # "vice First Last, Jr." - handles names with comma-separated suffixes but no reason
            r"vice\s+([A-Z][a-zA-Z]*(?:\s+[A-Z]\.?)*\s+[A-Z][a-zA-Z]+,\s+(?:Jr|Sr|III|II|IV)\.?)(?:\.|$)",
            # "vice Name reason" (no comma between name and reason) - for specific reasons
            r"vice\s+([^\s]+(?:\s+[^\s]+)*(?:\s+(?:Jr|Sr|III|II|IV)\.?)?)\s+(retired|deceased|resigned|elevated)(?:\.|$)",
            # "vice Name (with possible suffix), reason" - handles names with Jr., Sr., etc.
            r"vice\s+([^,]+(?:\s+(?:Jr|Sr|III|II|IV)\.?)?),\s+(.+?)(?:\.|$)",
            # "vice Full Name." - handles names without explicit reason (IMPROVED PATTERN)
            r"vice\s+([A-Z][a-zA-Z]+(?:\s+[A-Z]\.?)*(?:\s+[A-Z][a-zA-Z]+)+(?:,\s+(?:Jr|Sr|III|II|IV)\.?)?)(?:\.|$)",
            # "vice First Middle Last" - handles names with middle initials/names (BROADER PATTERN)
            r"vice\s+([A-Z][a-zA-Z]*(?:\s+[A-Z]\.?\s*)*[A-Z][a-zA-Z]+)(?:\.|$)",
            # Just "vice Name" (with possible suffix) - fallback
            r"vice\s+([^,\.]+(?:,\s+(?:Jr|Sr|III|II|IV)\.?)?)(?:\.|$)",
        ]

        for pattern in vice_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                predecessor_name = match.group(1).strip()
                # Clean up trailing commas but preserve periods after suffixes like Jr., Sr., etc.
                # Only remove periods if they're not part of a suffix
                if not re.search(r"(?:Jr|Sr|III|II|IV)\.?$", predecessor_name, re.IGNORECASE):
                    predecessor_name = re.sub(r"[,\.]$", "", predecessor_name)
                else:
                    # For suffixes, just remove trailing commas but keep periods
                    predecessor_name = re.sub(r",$", "", predecessor_name)
                    # Ensure suffix has a period if it doesn't already
                    if re.search(r"(?:Jr|Sr|III|II|IV)$", predecessor_name, re.IGNORECASE):
                        predecessor_name = re.sub(
                            r"(Jr|Sr|III|II|IV)$", r"\1.", predecessor_name, flags=re.IGNORECASE
                        )

                self.predecessor_name = predecessor_name if predecessor_name else None
                if len(match.groups()) > 1 and match.group(2):
                    cleaned_reason = match.group(2).strip()
                    self.vacancy_reason = cleaned_reason if cleaned_reason else None
                break

    def _fallback_parse(self, description: str) -> None:
        """Fallback parsing for non-standard formats."""

        # Check if this looks like a valid nomination description
        has_name_pattern = re.search(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*,\s+of\s+", description)
        has_to_be_pattern = re.search(r"to\s+be\s+", description, re.IGNORECASE)

        if not has_name_pattern or not has_to_be_pattern:
            # This doesn't look like a valid nomination description
            self.parsing_confidence = "failed"
            # For failed cases, set all fields to empty strings to match test expectations
            self.nominee_name = ""
            self.location = ""
            self.position_title = ""
            self.court_name = ""
            self.term_info = ""
            self.predecessor_name = ""
            self.vacancy_reason = ""
            return

        self.parsing_confidence = "low"

        # Try to extract name (first part before comma)
        name_match = re.search(r"^([^,]+)", description)
        if name_match:
            self.nominee_name = name_match.group(1).strip()

        # Try to extract location
        location_match = re.search(r"of\s+(?:the\s+)?([^,]+)", description, re.IGNORECASE)
        if location_match:
            location = location_match.group(1).strip()
            # Clean up location - remove "the" prefix
            location = re.sub(r"^the\s+", "", location, flags=re.IGNORECASE)
            self.location = location

        # Try to extract position
        position_match = re.search(
            r"to\s+be\s+(.+?)(?:\s+for\s+|\s+of\s+|,|\.)", description, re.IGNORECASE
        )
        if position_match:
            position_text = position_match.group(1).strip()
            self._extract_position_and_court(position_text)

        # Try to extract vice clause
        self._extract_vice_clause(description)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy DataFrame integration."""
        return {
            "nominee_name": self.nominee_name or None,
            "nomination_of_or_from_location": self.location or None,
            "nomination_to_position_title": self.position_title or None,
            "nomination_to_court_name": self.court_name or None,
            "nomination_term_info": self.term_info or None,
            "nomination_predecessor_name": self.predecessor_name or None,
            "nomination_vacancy_reason": self.vacancy_reason or None,
            "nomination_is_list_nomination": self.is_list_nomination,
            "nomination_parsing_confidence": self.parsing_confidence,
            "nomination_multiple_nominees_count": len(self.multiple_nominees),
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

    return pd.DataFrame.from_dict(parsed_data, orient="index")


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
            result_df[f"parsed_{col}"] = parsed_df[col]
        else:
            result_df[col] = parsed_df[col]

    return result_df
