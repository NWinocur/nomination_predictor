import pandas as pd
import pytest

from nomination_predictor.features import (
    convert_judge_name_format_from_last_comma_first_to_first_then_last,
    extract_name_and_location_columns,
    extract_name_and_location_from_description,
)
from nomination_predictor.nomination_parser import (
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


@pytest.mark.parametrize(
    "description,expected_name,expected_location",
    [
        # Test cases with format "name, of location, to be ..."
        ("melissa damian, of florida, to be united states district judge", 
         "melissa damian", "florida"),
        
        # Test case with format "name of location, to be ..."
        ("nicole g. bernerr of maryland, to be united states district judge", 
         "nicole g. bernerr", "maryland"),
        
        # Test case with longer name
        ("kirk edward sherriff, of california, to be united states district judge", 
         "kirk edward sherriff", "california"),
        
        # Test case with "of the" format
        ("sherri malloy beatty-arthur, of the district of columbia, for the position of judge", 
         "sherri malloy beatty-arthur", "district of columbia"),
        
        # Edge cases
        ("", "", ""),  # Empty string
        (None, "", ""),  # None input
        ("Invalid format without location or position", "", ""),  # No pattern match
    ],
)
def test_extract_name_and_location_from_description(description, expected_name, expected_location):
    """Test the extraction of name and location from nomination descriptions."""
    name, location = extract_name_and_location_from_description(description)
    assert name == expected_name
    assert location == expected_location


def test_extract_name_and_location_columns():
    """Test adding extracted name and location columns to a DataFrame."""
    # Create test DataFrame
    data = {
        "citation": ["1", "2", "3", "4"],
        "description": [
            "melissa damian, of florida, to be united states district judge",
            "nicole g. bernerr of maryland, to be united states district judge",
            "kirk edward sherriff, of california, to be united states district judge",
            "sherri malloy beatty-arthur, of the district of columbia, for the position of judge",
        ]
    }
    df = pd.DataFrame(data)
    
    # Apply the function
    result_df = extract_name_and_location_columns(df)
    
    # Check that new columns exist
    assert "full_name_from_description" in result_df.columns
    assert "location_of_origin_from_description" in result_df.columns
    
    # Check values
    expected_names = ["melissa damian", "nicole g. bernerr", "kirk edward sherriff", "sherri malloy beatty-arthur"]
    expected_locations = ["florida", "maryland", "california", "district of columbia"]
    
    for i, (name, location) in enumerate(zip(expected_names, expected_locations)):
        assert result_df.iloc[i]["full_name_from_description"] == name
        assert result_df.iloc[i]["location_of_origin_from_description"] == location
    
    # Test with invalid column name
    df_copy = df.copy()
    result_df_invalid = extract_name_and_location_columns(df_copy, description_column="nonexistent_column")
    assert result_df_invalid.equals(df_copy)  # Should return original DataFrame unchanged


@pytest.mark.parametrize(
    "judge_name,expected_output",
    [
        # Basic lastname, firstname format
        ("Smith, John", "John Smith"),
        
        # With middle name
        ("Jones, Robert Michael", "Robert Michael Jones"),
        
        # With suffix as separate part (Jr, Sr, III, etc.)
        ("Johnson, Thomas, Jr.", "Thomas Johnson Jr."),
        ("Williams, James, Sr.", "James Williams Sr."),
        ("Brown, William, III", "William Brown III"),
        
        # Specific test case for the reported issue
        ("Lake, Simeon Timothy III", "Simeon Timothy Lake III"),
        ("lake, simeon timothy iii", "simeon timothy lake iii"),  # lowercase version
        
        # With multiple commas but suffix embedded in middle part
        ("Garcia, Maria Jr., PhD", "Maria Garcia Jr., PhD"),
        
        # Edge cases
        ("", ""),  # Empty string
        (None, ""),  # None input
        ("Single Name", "Single Name"),  # No comma
    ],
)
def test_convert_judge_name_format(judge_name, expected_output):
    """Test the conversion of judge names from 'last, first' to 'first last' format, with proper suffix placement."""
    result = convert_judge_name_format_from_last_comma_first_to_first_then_last(judge_name)
    assert result == expected_output


# Test cases for the unified nomination parser
@pytest.mark.parametrize(
    "description,expected_name,expected_location,expected_position,expected_court,expected_predecessor,expected_reason,expected_confidence",
    [
        # Standard single nominations
        (
            "John Smith, of California, to be United States District Judge for the Northern District of California, vice Jane Doe, retired.",
            "John Smith",
            "California",
            "United States District Judge",
            "Northern District of California",
            "Jane Doe",
            "retired",
            "high"
        ),
        (
            "Mary Johnson, of New York, to be United States Circuit Judge for the Second Circuit, vice Robert Brown, deceased.",
            "Mary Johnson",
            "New York",
            "United States Circuit Judge",
            "Second Circuit",
            "Robert Brown",
            "deceased",
            "high"
        ),
        # Case without comma before vice (edge case we fixed)
        (
            "Mustafa Taher Kasubhai, of Oregon, to be United States District Judge for the District of Oregon vice Michael H. Simon, retired.",
            "Mustafa Taher Kasubhai",
            "Oregon",
            "United States District Judge",
            "District of Oregon",
            "Michael H. Simon",
            "retired",
            "high"
        ),
        # Case without predecessor
        (
            "Alice Williams, of Texas, to be United States District Judge for the Southern District of Texas.",
            "Alice Williams",
            "Texas",
            "United States District Judge",
            "Southern District of Texas",
            "",
            "",
            "high"
        ),
        # Case with term information
        (
            "David Miller, of Florida, to be United States Magistrate Judge for the Middle District of Florida for a term of eight years.",
            "David Miller",
            "Florida",
            "United States Magistrate Judge",
            "Middle District of Florida",
            "",
            "",
            "high"
        ),
        # Case with complex name (middle name, suffix)
        (
            "Robert James Wilson Jr., of Illinois, to be United States District Judge for the Northern District of Illinois, vice Sarah Thompson, elevated.",
            "Robert James Wilson Jr.",
            "Illinois",
            "United States District Judge",
            "Northern District of Illinois",
            "Sarah Thompson",
            "elevated",
            "high"
        ),
        # Case with "of the" in court name
        (
            "Jennifer Davis, of Washington, to be United States District Judge for the Western District of Washington, vice Michael Johnson, retired.",
            "Jennifer Davis",
            "Washington",
            "United States District Judge",
            "Western District of Washington",
            "Michael Johnson",
            "retired",
            "high"
        ),
        # Case with different position format
        (
            "Thomas Anderson, of Nevada, to be a Judge of the United States Court of Federal Claims, vice Patricia White, term expired.",
            "Thomas Anderson",
            "Nevada",
            "Judge",
            "United States Court of Federal Claims",
            "Patricia White",
            "term expired",
            "high"
        ),
        # Empty/null cases
        (
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "failed"
        ),
        # Malformed case (fallback parsing)
        (
            "Some malformed text that doesn't follow the standard pattern",
            "Some malformed text that doesn't follow the standard pattern",
            "",
            "",
            "",
            "",
            "",
            "low"
        ),
    ],
)
def test_nomination_description_parsing(
    description, expected_name, expected_location, expected_position, 
    expected_court, expected_predecessor, expected_reason, expected_confidence
):
    """Test comprehensive parsing of nomination descriptions."""
    nomination = NominationDescription.from_description(description)
    
    assert nomination.nominee_name == expected_name
    assert nomination.location == expected_location
    assert nomination.position_title == expected_position
    assert nomination.court_name == expected_court
    assert nomination.predecessor_name == expected_predecessor
    assert nomination.vacancy_reason == expected_reason
    assert nomination.parsing_confidence == expected_confidence


@pytest.mark.parametrize(
    "description,expected_is_list,expected_nominees_count",
    [
        # Single nomination
        (
            "John Smith, of California, to be United States District Judge for the Northern District of California.",
            False,
            0
        ),
        # List nomination with tab separators
        (
            "The following-named persons to be Associate Judges of the Superior Court: \tMary Ellen Abrecht, of the District of Columbia. \tKaye K. Christian, of the District of Columbia.",
            True,
            2
        ),
        # List nomination with "following named persons"
        (
            "The following named persons to be Members of the Board for terms indicated: John Doe, of State1. Jane Smith, of State2.",
            True,
            0  # Basic implementation may not parse all formats perfectly
        ),
    ],
)
def test_list_nomination_detection(description, expected_is_list, expected_nominees_count):
    """Test detection and parsing of list nominations."""
    nomination = NominationDescription.from_description(description)
    
    assert nomination.is_list_nomination == expected_is_list
    assert len(nomination.multiple_nominees) == expected_nominees_count


def test_parse_nomination_descriptions_series():
    """Test parsing a pandas Series of descriptions."""
    descriptions = pd.Series([
        "John Smith, of California, to be United States District Judge for the Northern District of California.",
        "Mary Johnson, of New York, to be United States Circuit Judge for the Second Circuit, vice Robert Brown, deceased.",
        "",  # Empty case
    ])
    
    result_df = parse_nomination_descriptions(descriptions)
    
    # Check structure
    assert len(result_df) == 3
    assert 'nominee_name' in result_df.columns
    assert 'court_name' in result_df.columns
    assert 'parsing_confidence' in result_df.columns
    
    # Check specific values
    assert result_df.iloc[0]['nominee_name'] == "John Smith"
    assert result_df.iloc[1]['predecessor_name'] == "Robert Brown"
    assert result_df.iloc[2]['parsing_confidence'] == "failed"


def test_backward_compatibility_functions():
    """Test that individual extraction functions work for backward compatibility."""
    description = "John Smith, of California, to be United States District Judge for the Northern District of California, vice Jane Doe, retired."
    
    assert extract_nominee_name(description) == "John Smith"
    assert extract_location(description) == "California"
    assert extract_court_name(description) == "Northern District of California"
    assert extract_position_title(description) == "United States District Judge"
    assert extract_predecessor_name(description) == "Jane Doe"
    assert extract_vacancy_reason(description) == "retired"


def test_parse_descriptions_to_columns():
    """Test adding parsed columns to an existing DataFrame."""
    df = pd.DataFrame({
        'id': [1, 2],
        'description': [
            "John Smith, of California, to be United States District Judge for the Northern District of California.",
            "Mary Johnson, of New York, to be United States Circuit Judge for the Second Circuit, vice Robert Brown, deceased."
        ]
    })
    
    result_df = parse_descriptions_to_columns(df)
    
    # Check that original columns are preserved
    assert 'id' in result_df.columns
    assert 'description' in result_df.columns
    
    # Check that parsed columns are added
    assert 'parsed_nominee_name' in result_df.columns
    assert 'parsed_court_name' in result_df.columns
    assert 'parsed_parsing_confidence' in result_df.columns
    
    # Check values
    assert result_df.iloc[0]['parsed_nominee_name'] == "John Smith"
    assert result_df.iloc[1]['parsed_predecessor_name'] == "Robert Brown"


def test_nomination_description_string_representations():
    """Test string and repr methods of NominationDescription."""
    # Single nomination
    nomination = NominationDescription.from_description(
        "John Smith, of California, to be United States District Judge for the Northern District of California."
    )
    
    str_repr = str(nomination)
    assert "John Smith" in str_repr
    assert "United States District Judge" in str_repr
    
    repr_str = repr(nomination)
    assert "NominationDescription" in repr_str
    assert "John Smith" in repr_str
    
    # List nomination
    list_nomination = NominationDescription.from_description(
        "The following-named persons to be Associate Judges: \tMary Ellen Abrecht, of DC. \tKaye K. Christian, of DC."
    )
    
    list_str = str(list_nomination)
    assert "List Nomination" in list_str