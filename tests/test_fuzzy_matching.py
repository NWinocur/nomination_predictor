"""Tests for fuzzy matching pipeline in features.py."""

import pandas as pd
import pytest

from nomination_predictor.fuzzy_matching import (
    calculate_court_similarity,
    calculate_date_similarity,
    calculate_name_similarity,
    find_matches_with_blocking,
    normalize_date_format,
)


def test_normalize_date_format():
    """Test date normalization function with various date formats."""
    # Test with different date formats
    assert normalize_date_format("2022-03-15").date() == pd.Timestamp("2022-03-15").date()
    assert normalize_date_format("03/15/2022").date() == pd.Timestamp("2022-03-15").date()
    assert normalize_date_format("2022/03/15").date() == pd.Timestamp("2022-03-15").date()
    
    # Test with datetime strings
    assert normalize_date_format("2022-03-15 14:30:00").date() == pd.Timestamp("2022-03-15").date()
    
    # Test with None/NaN
    assert normalize_date_format(None) is None
    assert normalize_date_format(pd.NA) is None
    
    # Test with invalid format
    assert normalize_date_format("not-a-date") is None


def test_calculate_name_similarity():
    """Test name similarity calculation with different name variations."""
    # Test exact match
    assert calculate_name_similarity("John Smith", "John Smith") == 100
    
    # Test name order variations
    assert calculate_name_similarity("John Smith", "Smith, John") >= 90
    
    # Test with middle names/initials
    assert calculate_name_similarity("John A. Smith", "John Smith") >= 90
    assert calculate_name_similarity("John Adam Smith", "John A. Smith") >= 85
    
    # Test with typos
    assert calculate_name_similarity("John Smith", "John Smtih") >= 80
    
    # Test with missing data
    assert calculate_name_similarity("John Smith", None) == 0
    assert calculate_name_similarity(None, "John Smith") == 0
    assert calculate_name_similarity("", "") == 0


def test_calculate_court_similarity():
    """Test court similarity calculation with variations in court names."""
    # Test exact match
    assert calculate_court_similarity(
        "United States District Court for the Southern District of New York", 
        "United States District Court for the Southern District of New York"
    ) == 100
    
    # Test with abbreviations
    assert calculate_court_similarity(
        "U.S. District Court, Southern District of New York",
        "United States District Court for the Southern District of New York"
    ) >= 90
    
    # Test with word order differences
    assert calculate_court_similarity(
        "Southern District of New York, U.S. District Court",
        "United States District Court for the Southern District of New York"
    ) >= 85
    
    # Test with missing data
    assert calculate_court_similarity("U.S. District Court", None) == 0
    assert calculate_court_similarity(None, "U.S. District Court") == 0
    assert calculate_court_similarity("", "") == 0


def test_calculate_date_similarity():
    """Test date similarity calculation based on proximity of dates."""
    # Test exact match
    assert calculate_date_similarity("2022-03-15", "2022-03-15") == 100
    
    # Test close dates (within window)
    assert calculate_date_similarity("2022-03-15", "2022-03-20") > 80
    assert calculate_date_similarity("2022-03-15", "2022-03-30") > 50
    
    # Test dates outside max window (default 45 days)
    assert calculate_date_similarity("2022-03-15", "2022-05-15") == 0
    
    # Test with custom window
    assert calculate_date_similarity("2022-03-15", "2022-05-15", max_days_diff=90) > 0
    
    # Test with missing data
    assert calculate_date_similarity("2022-03-15", None) == 0
    assert calculate_date_similarity(None, "2022-03-15") == 0


def test_find_matches_with_blocking():
    """Test the full fuzzy matching pipeline with sample datasets."""
    # Create sample Congress.gov data
    cong_data = [
        {
            "citation": "PN100",
            "full_name": "John A. Smith",
            "court_name": "U.S. District Court, Southern District of New York",
            "receiveddate": "2022-03-15",
            "last_name_from_full_name": "Smith"
        },
        {
            "citation": "PN200",
            "full_name": "Mary Jones",
            "court_name": "Northern District of California",
            "receiveddate": "2022-04-20",
            "last_name_from_full_name": "Jones"
        }
    ]
    
    # Create sample FJC data
    fjc_data = [
        {
            "nid": "FJC001",
            "name": "Smith, John Adam",
            "court": "Southern District of New York",
            "nomination_date": "2022-03-16",
            "last_name": "Smith"
        },
        {
            "nid": "FJC002",
            "name": "Jones, Mary Elizabeth",
            "court": "Northern District of California",
            "nomination_date": "2022-04-19",
            "last_name": "Jones"
        },
        {
            "nid": "FJC003",
            "name": "Wilson, Robert",
            "court": "Eastern District of Texas",
            "nomination_date": "2022-05-01",
            "last_name": "Wilson"
        }
    ]
    
    cong_df = pd.DataFrame(cong_data)
    fjc_df = pd.DataFrame(fjc_data)
    
    # Run matching
    results = find_matches_with_blocking(
        cong_df, 
        fjc_df,
        threshold=70  # Lower threshold for testing
    )
    
    # Verify results
    assert len(results) == len(cong_df)
    assert results.loc[0, 'match_score'] >= 70  # First record should match
    assert results.loc[1, 'match_score'] >= 70  # Second record should match
    
    # Verify correct matching
    assert results.loc[0, 'citation'] == "PN100"
    assert results.loc[0, 'nid'] == "FJC001"
    assert results.loc[1, 'citation'] == "PN200" 
    assert results.loc[1, 'nid'] == "FJC002"


def test_no_matches():
    """Test case where no matches should be found."""
    # Create sample Congress.gov data with no matches in FJC
    cong_data = [
        {
            "citation": "PN300",
            "full_name": "Unique Person",
            "court_name": "District of Alaska",
            "receiveddate": "2022-06-15",
            "last_name_from_full_name": "Person"
        }
    ]
    
    # Create sample FJC data with no matches for Congress data
    fjc_data = [
        {
            "nid": "FJC003",
            "name": "Different, Person",
            "court": "Western District of Washington",
            "nomination_date": "2022-01-16",
            "last_name": "Person"
        }
    ]
    
    cong_df = pd.DataFrame(cong_data)
    fjc_df = pd.DataFrame(fjc_data)
    
    # Run matching
    results = find_matches_with_blocking(
        cong_df, 
        fjc_df,
        threshold=70
    )
    
    # Verify no matches above threshold
    assert len(results) == 1
    assert results.loc[0, 'match_score'] < 70
