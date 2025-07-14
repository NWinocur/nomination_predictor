import pandas as pd
import pytest

from nomination_predictor.features import (
    convert_judge_name_format_from_last_comma_first_to_first_then_last,
    extract_name_and_location_columns,
    extract_name_and_location_from_description,
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
