"""
Unit tests for the fjc_data module.
"""

from datetime import datetime
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from nomination_predictor.fjc_data import (
    build_seat_timeline,
    crosswalk_congress_api,
    get_predecessor_info,
    load_fjc_csv,
    parse_fjc_date,
)


class TestFjcDateParsing(unittest.TestCase):
    """Test the FJC date parsing functions."""

    def test_parse_empty_date(self):
        """Test parsing empty date values."""
        self.assertTrue(pd.isna(parse_fjc_date("")))
        self.assertTrue(pd.isna(parse_fjc_date(None)))
        self.assertTrue(pd.isna(parse_fjc_date(np.nan)))
    
    def test_parse_excel_format_date(self):
        """Test parsing yyyy-mm-dd format (used by Excel for pre-1900)."""
        date = parse_fjc_date("1889-03-15")
        self.assertEqual(date.year, 1889)
        self.assertEqual(date.month, 3)
        self.assertEqual(date.day, 15)
        
        # Test edge case - single digit month and day
        date = parse_fjc_date("1889-3-5")
        self.assertEqual(date.year, 1889)
        self.assertEqual(date.month, 3)
        self.assertEqual(date.day, 5)
    
    def test_parse_csv_format_date(self):
        """Test parsing mm/dd/yyyy format (used in CSVs)."""
        date = parse_fjc_date("03/15/1889")
        self.assertEqual(date.year, 1889)
        self.assertEqual(date.month, 3)
        self.assertEqual(date.day, 15)
        
        # Test edge case - single digit month and day
        date = parse_fjc_date("3/5/1889")
        self.assertEqual(date.year, 1889)
        self.assertEqual(date.month, 3)
        self.assertEqual(date.day, 5)
    
    def test_parse_invalid_date(self):
        """Test parsing invalid date formats."""
        self.assertTrue(pd.isna(parse_fjc_date("not a date")))
        self.assertTrue(pd.isna(parse_fjc_date("32/03/1889")))  # More months than in a day; also day-first
        self.assertTrue(pd.isna(parse_fjc_date("1889-03-32")))  # More days than in a month
        self.assertTrue(pd.isna(parse_fjc_date("1889-13-15")))  # More months than in a year


@patch('nomination_predictor.fjc_data.FJC_DATA_DIR')
class TestLoadFjcCsv(unittest.TestCase):
    """Test loading FJC CSV files."""
    
    def test_load_csv_with_dates(self, mock_dir):
        """Test loading a CSV file with date columns."""
        # Create a mock CSV file path
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_dir.__truediv__.return_value = mock_path
        
        # Mock the pandas read_csv function
        mock_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'birth_date': ['1950-01-01', '1960-02-02', '1970-03-03'],
            'commission_date': ['01/01/1980', '02/02/1990', '03/03/2000']
        })
        
        with patch('pandas.read_csv', return_value=mock_df):
            result = load_fjc_csv('test.csv')
            
            # Check that date columns were properly parsed
            self.assertIsInstance(result['birth_date'].iloc[0], pd.Timestamp)
            self.assertIsInstance(result['commission_date'].iloc[0], pd.Timestamp)
            
            # Check values
            self.assertEqual(result['birth_date'].iloc[0].year, 1950)
            self.assertEqual(result['commission_date'].iloc[0].year, 1980)
    
    def test_file_not_found(self, mock_dir):
        """Test behavior when file doesn't exist."""
        # Create a mock file path that doesn't exist
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_dir.__truediv__.return_value = mock_path
        
        with self.assertRaises(FileNotFoundError):
            load_fjc_csv('nonexistent.csv')


class TestBuildSeatTimeline(unittest.TestCase):
    """Test building the seat timeline table."""
    
    def test_basic_seat_timeline(self):
        """Test basic seat timeline construction."""
        # Create sample data
        service_df = pd.DataFrame({
            'nid': ['A', 'B', 'C', 'D'],
            'seat_id': ['S1', 'S1', 'S2', 'S2'],
            'court': ['Court1', 'Court1', 'Court2', 'Court2'],
            'commission_date': [
                pd.Timestamp('1990-01-01'),
                pd.Timestamp('2000-01-01'),
                pd.Timestamp('1995-01-01'),
                pd.Timestamp('2005-01-01')
            ],
            'termination_date': [
                pd.Timestamp('1999-12-31'),
                pd.NaT,
                pd.Timestamp('2004-12-31'),
                pd.NaT
            ],
            'termination_reason': [
                'resignation',
                None,
                'retirement',
                None
            ]
        })
        
        timeline = build_seat_timeline(service_df)
        
        # Check that we have 4 rows
        self.assertEqual(len(timeline), 4)
        
        # Check that vacancy_date equals termination_date
        pd.testing.assert_series_equal(
            timeline['vacancy_date'].dropna(),
            timeline['termination_date'].dropna(),
            check_names=False  # Ignore Series names when comparing
        )
    
    def test_same_court_appointment_handling(self):
        """Test handling of 'appointment to same court' termination reason."""
        # Create sample data with 'appointment to same court'
        service_df = pd.DataFrame({
            'nid': ['A', 'B'],
            'seat_id': ['S1', 'S1'],
            'court': ['Court1', 'Court1'],
            'commission_date': [
                pd.Timestamp('1990-01-01'),
                pd.Timestamp('2000-01-01')
            ],
            'termination_date': [
                pd.Timestamp('1999-12-31'),
                pd.NaT
            ],
            'termination_reason': [
                'appointment to same court',
                None
            ]
        })
        
        timeline = build_seat_timeline(service_df)
        
        # Check that vacancy_date is NaT for the 'appointment to same court' row
        self.assertTrue(pd.isna(timeline.iloc[0]['vacancy_date']))
    
    def test_termination_date_fix(self):
        """Test fixing termination date > successor commission."""
        # Create sample data with inconsistent dates
        service_df = pd.DataFrame({
            'nid': ['A', 'B'],
            'seat_id': ['S1', 'S1'],
            'court': ['Court1', 'Court1'],
            'commission_date': [
                pd.Timestamp('1990-01-01'),
                pd.Timestamp('2000-01-01')
            ],
            'termination_date': [
                pd.Timestamp('2000-06-30'),  # After successor's commission
                pd.NaT
            ],
            'termination_reason': [
                'resignation',
                None
            ]
        })
        
        timeline = build_seat_timeline(service_df)
        
        # Check that termination_date was fixed to match successor's commission
        self.assertEqual(
            timeline.iloc[0]['termination_date'],
            pd.Timestamp('2000-01-01')
        )
        self.assertEqual(
            timeline.iloc[0]['vacancy_date'],
            pd.Timestamp('2000-01-01')
        )


class TestPredecessorInfo(unittest.TestCase):
    """Test getting predecessor info for crosswalking."""
    
    def test_get_predecessor_info(self):
        """Test getting predecessor information."""
        # Create sample seat timeline
        seat_timeline = pd.DataFrame({
            'seat_id': ['S1', 'S1', 'S2'],
            'court': ['Court1', 'Court1', 'Court2'],
            'nid': ['A', 'B', 'C'],
            'commission_date': [
                pd.Timestamp('1990-01-01'),
                pd.Timestamp('2000-01-01'),
                pd.Timestamp('1995-01-01')
            ],
            'termination_date': [
                pd.Timestamp('1999-12-31'),
                pd.NaT,
                pd.Timestamp('2004-12-31')
            ],
            'vacancy_date': [
                pd.Timestamp('1999-12-31'),
                pd.NaT,
                pd.Timestamp('2004-12-31')
            ]
        })
        
        lookup = get_predecessor_info(seat_timeline)
        
        # Check lookup contains only rows with vacancy dates
        self.assertEqual(len(lookup), 2)
        
        # Check columns and renaming
        self.assertIn('predecessor_nid', lookup.columns)
        self.assertEqual(lookup.iloc[0]['predecessor_nid'], 'A')
        self.assertEqual(lookup.iloc[1]['predecessor_nid'], 'C')


class TestCrosswalkCongressApi(unittest.TestCase):
    """Test crosswalking Congress.gov API data to FJC seat timeline."""
    
    def test_crosswalk_with_predecessor_name(self):
        """Test crosswalk with predecessor name in description."""
        # Create sample nomination data
        nomination_data = [
            {
                'description': 'Jane Smith, of California, to be Judge, vice John Doe.',
                'nomination_date': '2000-02-01'
            },
            {
                'description': 'Bob Johnson, of New York, to be Judge, succeeding Mary Brown.',
                'nomination_date': '2005-03-01'
            }
        ]
        
        # Create sample seat timeline
        seat_timeline = pd.DataFrame({
            'nid': ['JD', 'MB'],
            'seat_id': ['S1', 'S2'],
            'court': ['Court1', 'Court2'],
            'commission_date': [
                pd.Timestamp('1990-01-01'),
                pd.Timestamp('1995-01-01')
            ],
            'termination_date': [
                pd.Timestamp('1999-12-31'),
                pd.Timestamp('2004-12-31')
            ],
            'vacancy_date': [
                pd.Timestamp('1999-12-31'),
                pd.Timestamp('2004-12-31')
            ]
        })
        
        # Create sample judges data
        judges_df = pd.DataFrame({
            'nid': ['JD', 'MB'],
            'first_name': ['John', 'Mary'],
            'last_name': ['Doe', 'Brown']
        })
        
        result = crosswalk_congress_api(nomination_data, seat_timeline, judges_df)
        
        # Check that nominations were matched to seat_ids
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['seat_id'], 'S1')
        self.assertEqual(result.iloc[1]['seat_id'], 'S2')
        
        # Check match method
        self.assertEqual(result.iloc[0]['seat_match_method'], 'predecessor_name')
        self.assertEqual(result.iloc[1]['seat_match_method'], 'predecessor_name')


if __name__ == '__main__':
    unittest.main()
