"""Tests for the dataset module."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

# Import the module to test
from nomination_predictor.dataset import (
    DataPipelineError,
    check_id_uniqueness,
    parse_circuit_court,
    records_to_dataframe,
    save_dataframe_to_csv,
    validate_dataframe_ids,
)


def validate_vacancy_record(record: Dict[str, Any]) -> None:
    """Validate the structure and content of a vacancy record.
    
    Args:
        record: The record to validate
        
    Raises:
        AssertionError: If the record is invalid
    """
    assert isinstance(record, dict), "Record should be a dictionary"
    
    # Define required fields and their expected types
    required_field_types = {
        'court': str,
        'vacancy_date': str, # directly from our Internet-hosted data source this is presented to us as a string.  Conversion to datetime can happen during data cleaning & transformations.
        'vacancy_reason': str,
    }
    
    # Check required fields exist and have correct types
    for field, field_type in required_field_types.items():
        assert field in record, f"Missing required field: {field}"
        assert isinstance(record[field], field_type), \
            f"Field '{field}' has incorrect type. Expected {field_type}, got {type(record[field])}"
    
    # Optional fields (may be missing or empty)
    optional_fields = {
        'nominee': str,
        'nomination_date': str,
        'confirmation_date': str,
        'incumbent': str  # Now optional
    }
    
    for field, field_type in optional_fields.items():
        if field in record and record[field]:  # Only validate if field exists and is non-empty
            assert isinstance(record[field], field_type), \
                f"Field '{field}' has incorrect type. Expected {field_type}, got {type(record[field])}"
    
    # Validate date formats if present
    date_fields = ['vacancy_date', 'nomination_date', 'confirmation_date']
    for date_field in date_fields:
        if date_field in record and record[date_field]:
            try:
                datetime.strptime(record[date_field], '%m/%d/%Y')
            except ValueError:
                assert False, f"Invalid date format in field '{date_field}': {record[date_field]}. Expected MM/DD/YYYY"


# Fixtures
@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "seat": [1, 2],
            "court": ["9th Circuit", "DC Circuit"],
            "vacancy_date": ["2025-01-15", "2025-02-20"],
        }
    )


@pytest.fixture
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def year_archive_paths(fixtures_dir):
    """Return a list of paths to year archive HTML files."""
    pages_dir = fixtures_dir / "pages"
    return list(pages_dir.glob("archive_*.html"))


# Tests
def test_records_to_dataframe(sample_dataframe):
    """Test conversion of records to DataFrame."""
    records = [
        {"seat": "1", "court": "9th Circuit", "vacancy_date": "2025-01-15"},
        {"seat": "2", "court": "DC Circuit", "vacancy_date": "2025-02-20"},
    ]
    df = records_to_dataframe(records)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["seat", "court", "vacancy_date"]
    assert df["seat"].dtype == "int64"


def test_save_dataframe_to_csv(tmp_path, sample_dataframe):
    """Test saving DataFrame to CSV with pipe delimiter."""
    test_file = tmp_path / "test_output.csv"

    # Test successful save
    save_dataframe_to_csv(sample_dataframe, "test_output", tmp_path)
    assert test_file.exists()

    # Verify content with pipe delimiter
    df = pd.read_csv(test_file, sep='|')
    assert len(df) == 2
    assert list(df.columns) == ["seat", "court", "vacancy_date"]


# Tests for parse_circuit_court function
def test_parse_circuit_court_standard():
    """Test parsing standard circuit/court format."""
    # Standard circuit and district
    circuit, court = parse_circuit_court("09 - CA-N")
    assert circuit == 9
    assert court == "CA-N"
    
    # Circuit only (no district)
    circuit, court = parse_circuit_court("DC - DC")
    assert circuit is None
    assert court == "DC"
    
    # Federal circuit
    circuit, court = parse_circuit_court("03 - FED")
    assert circuit == 3
    assert court == "FED"


def test_parse_circuit_court_edge_cases():
    """Test edge cases and unusual but valid inputs."""
    # Single digit circuit
    circuit, court = parse_circuit_court("1 - CA")
    assert circuit == 1
    assert court == "CA"
    
    # Special court (no circuit)
    circuit, court = parse_circuit_court("IT")  # International Trade
    assert circuit is None
    assert court == "IT"
    
    # Federal Claims
    circuit, court = parse_circuit_court("CL")
    assert circuit is None
    assert court == "CL"


def test_parse_circuit_court_invalid():
    """Test handling of invalid inputs."""
    # Invalid formats
    with pytest.raises(ValueError):
        parse_circuit_court("")
    
    with pytest.raises(ValueError):
        parse_circuit_court(None)
    
    with pytest.raises(ValueError):
        parse_circuit_court("Invalid Format")
    
    with pytest.raises(ValueError):
        parse_circuit_court("XX - YY")  # Invalid circuit number


def test_parse_circuit_court_trailing_dash():
    """Test parsing court strings with trailing dashes."""
    from nomination_predictor.dataset import parse_circuit_court
    
    # Test with trailing dash after special court code
    circuit, court = parse_circuit_court("IT -")
    assert circuit is None
    assert court == "IT"
    
    # Test with trailing dash and whitespace after special court code
    circuit, court = parse_circuit_court("CL - ")
    assert circuit is None
    assert court == "CL"
    
    # Test with regular court code that has trailing dash
    circuit, court = parse_circuit_court("09 - CA-")
    assert circuit == 9
    assert court == "CA-"


# Tests for ID uniqueness checking
@pytest.fixture
def sample_fjc_df():
    """Create a sample FJC DataFrame with 'nid' column."""
    return pd.DataFrame({
        'nid': ['123', '456', '789', '123', None],  # Duplicate '123' and one None
        'name': ['Judge A', 'Judge B', 'Judge C', 'Judge A2', 'Unknown'],
        'court': ['CA-N', 'CA-S', 'NY-E', 'CA-C', 'DC']
    })


@pytest.fixture
def sample_congress_df():
    """Create a sample Congress API DataFrame with 'citation' column."""
    return pd.DataFrame({
        'citation': ['PN123', 'PN456', 'PN789', 'PN123', None],  # Duplicate 'PN123' and one None
        'nominee_name': ['Person A', 'Person B', 'Person C', 'Person A2', 'Unknown'],
        'court': ['CA-N', 'CA-S', 'NY-E', 'CA-C', 'DC']
    })


@pytest.fixture
def sample_hybrid_df():
    """Create a DataFrame with both 'nid' and 'citation' columns."""
    return pd.DataFrame({
        'nid': ['123', '456', '789'],
        'citation': ['PN123', 'PN456', 'PN789'],
        'name': ['Judge A', 'Judge B', 'Judge C']
    })


@pytest.fixture
def sample_missing_id_df():
    """Create a DataFrame with neither 'nid' nor 'citation' columns."""
    return pd.DataFrame({
        'name': ['Judge A', 'Judge B', 'Judge C'],
        'court': ['CA-N', 'CA-S', 'NY-E']
    })


def test_check_id_uniqueness_with_duplicates(sample_fjc_df):
    """Test checking uniqueness when duplicates exist."""
    result = check_id_uniqueness(sample_fjc_df, 'nid')
    
    assert result['is_unique'] is False
    assert result['duplicate_count'] == 1
    assert '123' in result['duplicate_values']
    assert result['duplicate_values']['123'] == 2
    assert len(result['duplicate_rows']) == 2  # Two rows with nid '123'


def test_check_id_uniqueness_all_unique(sample_congress_df):
    """Test checking uniqueness when all values are unique."""
    # Remove the duplicate row for this test
    unique_df = sample_congress_df[~sample_congress_df['citation'].duplicated()].copy()
    
    result = check_id_uniqueness(unique_df, 'citation')
    
    assert result['is_unique'] is True
    assert result['duplicate_count'] == 0
    assert len(result['duplicate_values']) == 0
    assert len(result['duplicate_rows']) == 0


def test_check_id_uniqueness_column_not_found():
    """Test checking uniqueness when the column doesn't exist."""
    df = pd.DataFrame({'name': ['A', 'B', 'C']})
    
    result = check_id_uniqueness(df, 'nonexistent_column')
    
    assert result['is_unique'] is True  # Default to True when column not found
    assert result['duplicate_count'] == 0
    

def test_validate_dataframe_ids_fjc(sample_fjc_df):
    """Test validating IDs with FJC DataFrame."""
    dataframes = {'judges': sample_fjc_df}
    results = validate_dataframe_ids(dataframes)
    
    assert 'judges' in results
    assert results['judges']['is_unique'] is False  # Because of duplicate nid
    

def test_validate_dataframe_ids_congress(sample_congress_df):
    """Test validating IDs with Congress DataFrame."""
    dataframes = {'nominations': sample_congress_df}
    results = validate_dataframe_ids(dataframes)
    
    assert 'nominations' in results
    assert results['nominations']['is_unique'] is False  # Because of duplicate citation


def test_validate_dataframe_ids_multiple_dataframes(sample_fjc_df, sample_congress_df):
    """Test validating IDs across multiple DataFrames."""
    dataframes = {'judges': sample_fjc_df, 'nominations': sample_congress_df}
    results = validate_dataframe_ids(dataframes)
    
    assert 'judges' in results
    assert 'nominations' in results
    assert results['judges']['is_unique'] is False  # Duplicate nid
    assert results['nominations']['is_unique'] is False  # Duplicate citation
