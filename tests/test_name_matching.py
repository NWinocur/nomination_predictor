import pandas as pd
import pytest

from nomination_predictor.name_matching import extract_predecessor, fill_predecessor_column


@pytest.mark.parametrize(
    "description,expected_predecessor",
    [
        # Standard cases with retiring, elevated, deceased judges
        (", vice ann l. aiken, retiring", "ann l. aiken"),
        (", vice diana gribbon motz, elevated", "diana gribbon motz"),
        (", vice john adams, deceased", "john adams"),
        
        # Cases with different spacing or punctuation in names
        (", vice wendell p. gardner,jr., retired.", "wendell p. gardner,jr."),
        (", vice Thomas A. Smith, Jr., retired", "Thomas A. Smith, Jr."),
        
        # Cases with new positions
        (" vice a new position created by public law 101-650", "a new position created by public law 101-650"),
        (" vice a new position created by p. l. 101-650", "a new position created by public law 101-650"),
        (" vice a new position created by p.l. 98-253", "a new position created by public law 98-253"),
        
        # Edge cases
        ("", None),  # Empty string
        ("this description does not contain a word which starts with v and ends with ice", None),  # No "vice" in the description
        (None, None),  # None input
    ],
)
def test_extract_predecessor(description, expected_predecessor):
    """Test the extraction of predecessor name from nomination descriptions."""
    predecessor = extract_predecessor(description)
    assert predecessor == expected_predecessor


def test_fill_predecessor_column():
    """Test filling missing values in nominees_0_predecessorname column."""
    # Create test DataFrame
    data = {
        "nominees_0_predecessorname": ["", None, "Existing Name", "", None],
        "description": [
            ", vice ann l. aiken, retiring",
            ", vice diana gribbon motz, elevated",
            "This shouldn't change the existing name",
            "No vice information here",
            " vice a new position created by public law 101-650"
        ]
    }
    df = pd.DataFrame(data)
    
    # Apply the function
    result_df = fill_predecessor_column(df)
    
    # Check that values were filled correctly
    expected_values = [
        "ann l. aiken",
        "diana gribbon motz", 
        "Existing Name",  # Unchanged
        None,             # No vice info, should remain None
        "a new position created by public law 101-650"
    ]
    
    for i, expected in enumerate(expected_values):
        assert result_df.iloc[i]["nominees_0_predecessorname"] == expected


def test_normalize_public_law_references():
    """Test normalization of 'public law' references."""
    # This tests the specific normalization for new position cases
    test_cases = [
        "a new position created by public law 101-650",
        "a new position created by p. l. 101-650", 
        "a new position created by p.l. 98-253"
    ]
    
    expected = [
        "a new position created by public law 101-650",
        "a new position created by public law 101-650",
        "a new position created by public law 98-253"
    ]
    
    # Test each case individually
    for i, test_case in enumerate(test_cases):
        description = f" vice {test_case}"
        result = extract_predecessor(description)
        assert result == expected[i]
