#!/usr/bin/env python3
"""
Parametrized pytest tests for the unified nomination description parser.
"""

import re

import pandas as pd
import pytest

from nomination_predictor.nomination_description_parser import (
    NominationDescription,
    extract_court_name,
    extract_location,
    extract_nominee_name,
    extract_position_title,
    extract_predecessor_name,
    extract_vacancy_reason,
    parse_descriptions_to_columns,
    parse_nomination_descriptions,
)

# Test data for parametrized tests
SINGLE_NOMINATION_TEST_CASES = [
    pytest.param(
        "John Smith, of California, to be United States District Judge for the Northern District of California, vice Jane Doe, retired.",
        "John Smith",
        "California",
        "United States District Judge",
        "Northern District of California",
        "Jane Doe",
        "retired",
        "high",
        id="standard_district_judge_with_predecessor"
    ),
    pytest.param(
        "Mustafa Taher Kasubhai, of Oregon, to be United States District Judge for the District of Oregon vice Michael H. Simon, retired.",
        "Mustafa Taher Kasubhai",
        "Oregon",
        "United States District Judge",
        "District of Oregon",
        "Michael H. Simon",
        "retired",
        "high",
        id="missing_comma_before_vice"
    ),
    pytest.param(
        "Alice Williams, of Texas, to be United States District Judge for the Southern District of Texas.",
        "Alice Williams",
        "Texas",
        "United States District Judge",
        "Southern District of Texas",
        "",
        "",
        "high",
        id="no_predecessor"
    ),
    pytest.param(
        "Mary Johnson, of New York, to be United States Circuit Judge for the Second Circuit, vice Robert Brown, deceased.",
        "Mary Johnson",
        "New York",
        "United States Circuit Judge",
        "Second Circuit",
        "Robert Brown",
        "deceased",
        "high",
        id="circuit_judge_deceased_predecessor"
    ),
    pytest.param(
        "Nicholas George Miranda, of the District of Columbia, to be an Associate Judge of the Superior Court of the District of Columbia for the term of fifteen years, vice Rupa Ranga Puttagunta, resigned.",
        "Nicholas George Miranda",
        "District of Columbia",
        "Associate Judge",
        "Superior Court of the District of Columbia",
        "Rupa Ranga Puttagunta",
        "resigned",
        "high",
        id="associate_judge"
    ),
#    pytest.param(
#        "James Allan Hurd, Jr., of the Virgin Islands, to be United States Attorney for the District of the Virgin Islands for the term of four years vice James W. Diehm, resigned.",
#        "James Allan Hurd, Jr.",
#        "Virgin Islands",
#        "United States Attorney",
#        "District of the Virgin Islands",
#        "James W. Diehm",
#        "resigned",
#        "high",
#        id="attorney"  # skipped because attorney positions are out of scope for project, and an upstream data-cleaning cell's bug has been fixed to filter them earlier
#    ),
    pytest.param(
        "Hugh Lawson, of Georgia, to be United States District Judge for the Middle District of Georgia vice Wilbur D. Owens, Jr., retired.",
        "Hugh Lawson",
        "Georgia",
        "United States District Judge",
        "Middle District of Georgia",
        "Wilbur D. Owens, Jr.",
        "retired",
        "high",
        id="district_judge_retired_predecessor"
    ),
    pytest.param(
        "David Miller, of Florida, to be United States Magistrate Judge for the Middle District of Florida for a term of eight years.",
        "David Miller",
        "Florida",
        "United States Magistrate Judge",
        "Middle District of Florida",
        None,
        None,
        "high",
        id="magistrate_judge_with_term"
    ),
    pytest.param(
        "Donald C. Pogue, of Connecticut, to be a Judge of the United States Court of International Trade vice James L. Watson, retired.",
        "Donald C. Pogue",
        "Connecticut",
        "Judge",
        "United States Court of International Trade",
        "James L. Watson",
        "retired",
        "high",
        id="judge_of_court_of_international_trade"
    ),
    pytest.param(
        "Lawrence Baskir, of Maryland, to be a Judge of the United States Court of Federal Claims for a term of fifteen years, vice Reginald W. Gibson, retired.",
        "Lawrence Baskir",
        "Maryland",
        "Judge",
        "United States Court of Federal Claims",
        "Reginald W. Gibson",
        "retired",
        "high",
        id="judge_of_court_of_federal_claims"
    ),
    pytest.param(
        "Anabelle Rodriguez, of Puerto Rico, to be United States District Judge for the District of Puerto Rico vice Raymond L. Acosta, retired.",
        "Anabelle Rodriguez",
        "Puerto Rico",
        "United States District Judge",
        "District of Puerto Rico",
        "Raymond L. Acosta",
        "retired",
        "high",
        id="district_judge"
    ),
    pytest.param(
        "David W. Hagen, of Nevada, to be United States District Judge for the District of Nevada vice Edward C. Reed, Jr., retired.",
        "David W. Hagen",
        "Nevada",
        "United States District Judge",
        "District of Nevada",
        "Edward C. Reed, Jr.",
        "retired",
        "high",
        id="district_judge_predecessor_jr_suffix"
    ),
    pytest.param(
        "Eric T. Washington, of the District of Columbia, to be an Associate Judge of the Superior Court of the District of Columbia for the term of fifteen years,] vice Ricardo M. Urbina, elevated.",
        "Eric T. Washington",
        "District of Columbia",
        "Associate Judge",
        "Superior Court of the District of Columbia",
        "Ricardo M. Urbina",
        "elevated",
        "high",
        id="associate_judge_unexpected_punctuation_elevated"
    ),
    pytest.param(
        "John C. Truong, of the District of Columbia, to be an Associate Judge of the Superior Court of the District of Columbia for the term of fifteen years, vice Wendell P. Gardner, Jr., retired.",
        "John C. Truong",
        "District of Columbia",
        "Associate Judge",
        "Superior Court of the District of Columbia",
        "Wendell P. Gardner, Jr.",
        "retired",
        "high",
        id="associate_judge_predecessor_retired"
    ),
    pytest.param(
        "Sandra Day O'Connor, of Arizona, to be an Associate Justice of the Supreme Court of the United States, vice Potter Stewart, retired.",
        "Sandra Day O'Connor",
        "Arizona",
        "Associate Justice",
        "Supreme Court of the United States",
        "Potter Stewart",
        "retired",
        "high",
        id="associate_justice_predecessor_retired"
    ),
    pytest.param(
        "Merrick B. Garland, of Maryland, to be an Associate Justice of the Supreme Court of the United States, vice Antonin Scalia, deceased.",
        "Merrick B. Garland",
        "Maryland",
        "Associate Justice",
        "Supreme Court of the United States",
        "Antonin Scalia",
        "deceased",
        "high",
        id="associate_justice_predecessor_deceased"
    ),
    pytest.param(
        "Samuel A. Alito, Jr., of New Jersey, to be an Associate Justice of the Supreme Court of the United States, vice Sandra Day O'Connor, retiring.",
        "Samuel A. Alito, Jr.",
        "New Jersey",
        "Associate Justice",
        "Supreme Court of the United States",
        "Sandra Day O'Connor",
        "retiring",
        "high",
        id="associate_justice_predecessor_retiring"
    ),
    pytest.param(
        "George W. Mitchell, of the District of Columbia, to be an Associate Judge of the Superior Court of the District of Columbia for a term of fifteen years, vice William E. Stewart, Jr., retired.",
        "George W. Mitchell",
        "District of Columbia",
        "Associate Judge",
        "Superior Court of the District of Columbia",
        "William E. Stewart, Jr.",
        "retired",
        "high",
        id="associate_judge_term_limit"
    ),
    pytest.param(
        "John A. Terry, of the District of Columbia, to be an Associate Judge of the District of Columbia Court of Appeals for the term of fifteen years, vice Stanley S. Harris.",
        "John A. Terry",
        "District of Columbia",
        "Associate Judge",
        "District of Columbia Court of Appeals",
        "Stanley S. Harris",
        None,
        "high",
        id="associate_appeals_reason_for_vacancy_missing"
    ),
    pytest.param(
        "Chester J. Straub, of New York,to be United States Circuit Judge for the Second Circuit, vice Joseph M. McLaughlin, retired.",
        "Chester J. Straub",
        "New York",
        "United States Circuit Judge",
        "Second Circuit",
        "Joseph M. McLaughlin",
        "retired",
        "high",
        id="circuit_judge_retired"
    ),
    pytest.param(
        "Sonia Sotomayor, of New York, to the United States District Judge for the Southern District of New York vice John M. Walker, Jr., elevated.",
        "Sonia Sotomayor",
        "New York",
        "United States District Judge",
        "Southern District of New York",
        "John M. Walker, Jr.",
        "elevated",
        "high",
        id="district_judge_elevated"
    ),
    pytest.param(
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "failed",
        id="empty_description"
    ),
    pytest.param(
        "Some malformed text that doesn't follow the standard pattern",
        "",
        "",
        "",
        "",
        "",
        "",
        "failed",
        id="malformed_description"
    ),
]

LIST_NOMINATION_TEST_CASES = [
    pytest.param(
        "The following-named persons to be Associate Judges of the Superior Court: \tMary Ellen Abrecht, of the District of Columbia. \tKaye K. Christian, of the District of Columbia.",
        True,
        2,
        "medium",
        id="tab_separated_list_nomination"
    ),
    pytest.param(
        "The following-named persons to be United States District Judges: John Smith, of California, and Jane Doe, of New York.",
        True,
        2,
        "medium",
        id="and_separated_list_nomination"
    ),
]

BACKWARD_COMPATIBILITY_TEST_CASES = [
    pytest.param(
        "John Smith, of California, to be United States District Judge for the Northern District of California, vice Jane Doe, retired.",
        "John Smith",
        "California",
        "Northern District of California",
        "United States District Judge",
        "Jane Doe",
        "retired",
        id="full_standard_nomination"
    ),
]
    
    
# Decomposed tests for better failure isolation and diagnostics

@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_nominee_name_extraction(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that nominee names are correctly extracted from nomination descriptions."""
    nomination = NominationDescription.from_description(description)
    assert nomination.nominee_name == expected_name


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_location_extraction(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that locations are correctly extracted from nomination descriptions."""
    nomination = NominationDescription.from_description(description)
    assert nomination.location == expected_location


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_position_title_extraction(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that position titles are correctly extracted from nomination descriptions."""
    nomination = NominationDescription.from_description(description)
    assert nomination.position_title == expected_position_title


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_court_name_extraction(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that court names are correctly extracted from nomination descriptions."""
    nomination = NominationDescription.from_description(description)
    assert nomination.court_name == expected_court


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_court_name_cleanliness(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that court names do not contain term information contamination."""
    nomination = NominationDescription.from_description(description)
    if nomination.court_name:  # Only test if court name exists
        assert all([term_keyword not in nomination.court_name for term_keyword in ["term of", "years"]])


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_predecessor_name_extraction(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that predecessor names are correctly extracted from nomination descriptions."""
    nomination = NominationDescription.from_description(description)
    assert nomination.predecessor_name == expected_predecessor


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_vacancy_reason_extraction(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that vacancy reasons are correctly extracted from nomination descriptions."""
    nomination = NominationDescription.from_description(description)
    assert nomination.vacancy_reason == expected_reason


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_vacancy_reason_cleanliness(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that vacancy reasons do not contain name suffix contamination."""
    nomination = NominationDescription.from_description(description)
    if nomination.vacancy_reason:  # Only test if vacancy reason exists
        assert all([suffix not in nomination.vacancy_reason for suffix in ["Jr.", "Sr.", "II", "III", "IV", "V"]])


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_parsing_confidence_assignment(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that parsing confidence levels are correctly assigned."""
    nomination = NominationDescription.from_description(description)
    assert nomination.parsing_confidence == expected_confidence


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_single_nomination_flag(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that single nominations are correctly identified as non-list nominations."""
    nomination = NominationDescription.from_description(description)
    assert not nomination.is_list_nomination


# Unit tests for internal parsing methods using known-good expected outputs

@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_extract_components_from_match_integration(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that _extract_components_from_match correctly processes regex matches."""
    nomination = NominationDescription()
    
    # Test the main parsing patterns to find a match
    patterns = [
        # Pattern with term and vice clause
        r'^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?)\s+for\s+(?:a\s+)?term\s+(?:of\s+)?(.+?),\s+vice\s+(.+?),\s+(.+?)\.?$',
        # Pattern with term but no vice clause
        r'^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?)\s+for\s+(?:a\s+)?term\s+(?:of\s+)?(.+?)\.?$',
        # Pattern with vice clause but no term
        r'^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?),\s+vice\s+(.+?),\s+(.+?)\.?$',
        # Pattern with vice clause but no explicit reason
        r'^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?)\s+vice\s+(.+?),\s+(.+?)\.?$',
        # Pattern with vice clause but no comma before vice
        r'^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?)\s+vice\s+(.+?)\s+(retired|deceased|resigned|elevated)\.?$',
        # Simple pattern without predecessor
        r'^(.+?),\s+of\s+(.+?),\s+to\s+be\s+(.+?)\.?$',
    ]
    
    match_found = False
    for pattern in patterns:
        match = re.search(pattern, description, re.IGNORECASE | re.DOTALL)
        if match:
            nomination._extract_components_from_match(match, description)
            match_found = True
            break
    
    # If no pattern matched, this test case might need fallback parsing
    if not match_found:
        nomination._fallback_parse(description)
    
    # Verify the components were extracted correctly
    assert nomination.nominee_name == expected_name
    assert nomination.location == expected_location
    assert nomination.position_title == expected_position_title
    assert nomination.court_name == expected_court
    assert nomination.predecessor_name == expected_predecessor
    assert nomination.vacancy_reason == expected_reason


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_extract_position_and_court_method(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that _extract_position_and_court correctly separates position titles from court names."""
    nomination = NominationDescription()
    
    # Extract the position text from the description (simulate what the main parser does)
    position_match = re.search(r'to\s+be\s+(.+?)(?:\s+for\s+(?:a\s+)?term\s+|,\s+vice\s+|\.)', description, re.IGNORECASE)
    if position_match:
        position_text = position_match.group(1).strip()
        nomination._extract_position_and_court(position_text)
        
        # Verify position and court extraction
        assert nomination.position_title == expected_position_title
        assert nomination.court_name == expected_court
        
        # Verify court name cleanliness (no term contamination)
        if nomination.court_name:
            assert all([term_keyword not in nomination.court_name for term_keyword in ["term of", "years"]])


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_extract_vice_clause_method(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that _extract_vice_clause correctly extracts predecessor and vacancy reason."""
    nomination = NominationDescription()
    
    # Only test cases that have a vice clause
    if "vice" in description.lower() and expected_predecessor:
        nomination._extract_vice_clause(description)
        
        # Verify predecessor and vacancy reason extraction
        assert nomination.predecessor_name == expected_predecessor
        assert nomination.vacancy_reason == expected_reason
        
        # Verify vacancy reason cleanliness (no suffix contamination)
        if nomination.vacancy_reason:
            assert all([suffix not in nomination.vacancy_reason for suffix in ["Jr.", "Sr.", "II", "III", "IV", "V"]])


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_clean_predecessor_name_method(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that _clean_predecessor_name properly handles name suffixes."""
    nomination = NominationDescription()
    
    # Only test cases that have a predecessor
    if expected_predecessor:
        # Extract raw predecessor name from description
        vice_match = re.search(r'vice\s+([^,]+(?:,\s+(?:Jr|Sr|III|II|IV)\.?)?)', description, re.IGNORECASE)
        if vice_match:
            raw_predecessor = vice_match.group(1).strip()
            cleaned_predecessor = nomination._clean_predecessor_name(raw_predecessor)
            
            # The cleaned predecessor should match our expected output
            assert cleaned_predecessor == expected_predecessor
            
            # Verify suffix handling (should have proper periods)
            if re.search(r'(?:Jr|Sr|III|II|IV)', cleaned_predecessor, re.IGNORECASE):
                assert cleaned_predecessor.endswith('.')


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position_title,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_clean_vacancy_reason_method(
    description, expected_name, expected_location, expected_position_title, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test that _clean_vacancy_reason removes suffix contamination."""
    nomination = NominationDescription()
    
    # Only test cases that have a vacancy reason
    if expected_reason:
        # Extract raw vacancy reason from description
        reason_match = re.search(r'vice\s+[^,]+,\s+(.+?)\.?$', description, re.IGNORECASE)
        if reason_match:
            raw_reason = reason_match.group(1).strip()
            cleaned_reason = nomination._clean_vacancy_reason(raw_reason)
            
            # The cleaned reason should match our expected output
            assert cleaned_reason == expected_reason
            
            # Verify no suffix contamination
            assert all([suffix not in cleaned_reason for suffix in ["Jr.", "Sr.", "II", "III", "IV", "V"]])


@pytest.mark.parametrize(
    "description,expected_is_list,expected_nominee_count,expected_confidence",
    LIST_NOMINATION_TEST_CASES
)
@pytest.mark.skip(reason="Multi-person nominations not yet supported because for judge nominations, no such instances existed, so there was no need")
def test_list_nomination_parsing(
    description, expected_is_list, expected_nominee_count, expected_confidence
):
    """Test parsing of list nomination descriptions."""
    nomination = NominationDescription.from_description(description)
    
    assert nomination.is_list_nomination == expected_is_list
    assert len(nomination.multiple_nominees) == expected_nominee_count
    assert nomination.parsing_confidence == expected_confidence


@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_court,expected_position,expected_predecessor,expected_reason",
    BACKWARD_COMPATIBILITY_TEST_CASES
)
def test_backward_compatibility_functions(
    description, expected_name, expected_location, expected_court, 
    expected_position, expected_predecessor, expected_reason
):
    """Test backward compatibility extraction functions."""
    assert extract_nominee_name(description) == expected_name
    assert extract_location(description) == expected_location
    assert extract_court_name(description) == expected_court
    assert extract_position_title(description) == expected_position
    assert extract_predecessor_name(description) == expected_predecessor
    assert extract_vacancy_reason(description) == expected_reason


def test_batch_processing_with_pandas():
    """Test batch processing of descriptions using pandas Series."""
    test_descriptions = [
        "John Smith, of California, to be United States District Judge for the Northern District of California, vice Jane Doe, retired.",
        "Mary Johnson, of New York, to be United States Circuit Judge for the Second Circuit, vice Robert Brown, deceased.",
        "Alice Williams, of Texas, to be United States District Judge for the Southern District of Texas.",
    ]
    
    descriptions_series = pd.Series(test_descriptions)
    parsed_df = parse_nomination_descriptions(descriptions_series)
    
    # Verify DataFrame structure
    assert len(parsed_df) == 3
    assert 'nominee_name' in parsed_df.columns
    assert 'nomination_to_court_name' in parsed_df.columns
    assert 'nomination_predecessor_name' in parsed_df.columns
    assert 'nomination_parsing_confidence' in parsed_df.columns
    
    # Verify specific parsing results
    assert parsed_df.iloc[0]['nominee_name'] == "John Smith"
    assert parsed_df.iloc[0]['nomination_to_court_name'] == "Northern District of California"
    assert parsed_df.iloc[1]['nomination_predecessor_name'] == "Robert Brown"
    assert parsed_df.iloc[2]['nomination_to_court_name'] == "Southern District of Texas"


def test_dataframe_integration():
    """Test integration with existing DataFrames."""
    sample_df = pd.DataFrame({
        'id': [1, 2, 3],
        'description': [
            "John Smith, of California, to be United States District Judge for the Northern District of California.",
            "Mary Johnson, of New York, to be United States Circuit Judge for the Second Circuit, vice Robert Brown, deceased.",
            "Alice Williams, of Texas, to be United States District Judge for the Southern District of Texas."
        ]
    })
    
    result_df = parse_descriptions_to_columns(sample_df)
    
    # Verify original columns are preserved
    assert 'id' in result_df.columns
    assert 'description' in result_df.columns
    
    # Verify parsed columns are added
    assert 'nominee_name' in result_df.columns
    assert 'nomination_to_court_name' in result_df.columns
    assert 'nomination_predecessor_name' in result_df.columns
    
    # Verify parsing results
    assert result_df.iloc[0]['nominee_name'] == "John Smith"
    assert result_df.iloc[1]['nomination_predecessor_name'] == "Robert Brown"
    assert result_df.iloc[2]['nomination_to_court_name'] == "Southern District of Texas"


def test_nomination_description_string_representation():
    """Test string representation methods of NominationDescription."""
    description = "John Smith, of California, to be United States District Judge for the Northern District of California, vice Jane Doe, retired."
    nomination = NominationDescription.from_description(description)
    
    # Test __str__ method
    str_repr = str(nomination)
    assert "John Smith" in str_repr
    assert "California" in str_repr
    
    # Test __repr__ method
    repr_str = repr(nomination)
    assert "NominationDescription" in repr_str
    assert "John Smith" in repr_str


def test_edge_cases():
    """Test various edge cases and error conditions."""
    # Test None input
    nomination = NominationDescription.from_description(None)
    assert nomination.parsing_confidence == "failed"
    
    # Test very short description
    nomination = NominationDescription.from_description("short")
    assert nomination.parsing_confidence == "failed"
    
    # Test description with unusual formatting
    unusual_desc = "JOHN SMITH, OF CALIFORNIA, TO BE UNITED STATES DISTRICT JUDGE"
    nomination = NominationDescription.from_description(unusual_desc)
    # Should still extract some information despite unusual formatting
    assert nomination.nominee_name or nomination.parsing_confidence in ["low", "failed"]


def test_confidence_levels():
    """Test that confidence levels are assigned appropriately."""
    # High confidence case
    high_conf_desc = "John Smith, of California, to be United States District Judge for the Northern District of California, vice Jane Doe, retired."
    nomination = NominationDescription.from_description(high_conf_desc)
    assert nomination.parsing_confidence == "high"
    
    # Failed case
    failed_desc = ""
    nomination = NominationDescription.from_description(failed_desc)
    assert nomination.parsing_confidence == "failed"
