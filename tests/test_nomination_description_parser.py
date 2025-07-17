#!/usr/bin/env python3
"""
Parametrized pytest tests for the unified nomination description parser.
"""

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
        "Second Circuit",
        "Robert Brown",
        "deceased",
        "high",
        id="circuit_judge_deceased_predecessor"
    ),
    pytest.param(
        "David Miller, of Florida, to be United States Magistrate Judge for the Middle District of Florida for a term of eight years.",
        "David Miller",
        "Florida",
        "Middle District of Florida",
        "",
        "",
        "high",
        id="magistrate_judge_with_term"
    ),
    pytest.param(
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
    
    
@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_court,expected_predecessor,expected_reason,expected_confidence",
    SINGLE_NOMINATION_TEST_CASES
)
def test_single_nomination_parsing(
    description, expected_name, expected_location, expected_court, 
    expected_predecessor, expected_reason, expected_confidence
):
    """Test parsing of single nomination descriptions."""
    nomination = NominationDescription.from_description(description)
    
    assert nomination.nominee_name == expected_name
    assert nomination.location == expected_location
    assert nomination.court_name == expected_court
    assert nomination.predecessor_name == expected_predecessor
    assert nomination.vacancy_reason == expected_reason
    assert nomination.parsing_confidence == expected_confidence
    assert not nomination.is_list_nomination


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
    assert 'court_name' in parsed_df.columns
    assert 'predecessor_name' in parsed_df.columns
    assert 'parsing_confidence' in parsed_df.columns
    
    # Verify specific parsing results
    assert parsed_df.iloc[0]['nominee_name'] == "John Smith"
    assert parsed_df.iloc[0]['court_name'] == "Northern District of California"
    assert parsed_df.iloc[1]['predecessor_name'] == "Robert Brown"
    assert parsed_df.iloc[2]['court_name'] == "Southern District of Texas"


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
    assert 'parsed_nominee_name' in result_df.columns
    assert 'parsed_court_name' in result_df.columns
    assert 'parsed_predecessor_name' in result_df.columns
    
    # Verify parsing results
    assert result_df.iloc[0]['parsed_nominee_name'] == "John Smith"
    assert result_df.iloc[1]['parsed_predecessor_name'] == "Robert Brown"
    assert result_df.iloc[2]['parsed_court_name'] == "Southern District of Texas"


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
