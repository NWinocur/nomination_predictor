#!/usr/bin/env python
"""
Script to demonstrate and test the FJC data processing functionality.

This script serves as both a demonstration and a practical test of the
fjc_data module's capabilities with actual FJC data.
"""

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add the project root to the path so we can import our module
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from nomination_predictor.fjc_data import (
    FJC_DATA_DIR,
    PROCESSED_DATA_DIR,
    build_seat_timeline,
    crosswalk_congress_api,
    get_predecessor_info,
    load_fjc_csv,
    parse_fjc_date,
)


def test_date_parsing():
    """Test date parsing functionality with examples."""
    print("Testing date parsing functionality...")
    
    test_dates = [
        "1889-03-15",  # Excel format
        "03/15/1889",  # CSV format
        "1/5/2000",    # Single digit month/day
        "",            # Empty
        None,          # None
        "invalid"      # Invalid
    ]
    
    for date_str in test_dates:
        parsed = parse_fjc_date(date_str)
        print(f"'{date_str}' -> {parsed}")
    
    print("Date parsing test complete.\n")


def test_load_csv():
    """Test loading an FJC CSV file."""
    print("Testing CSV loading functionality...")
    
    try:
        judges_df = load_fjc_csv('judges.csv')
        print(f"Successfully loaded judges.csv with {len(judges_df)} rows")
        print(f"Column count: {len(judges_df.columns)}")
        print(f"First few columns: {list(judges_df.columns)[:10]}")
        
        # Check date columns
        date_cols = [col for col in judges_df.columns if 'date' in col.lower()]
        print(f"Date columns: {date_cols}")
        
        for col in date_cols:
            non_na_count = judges_df[col].notna().sum()
            print(f"  {col}: {non_na_count} non-NA values")
            
            # Show a sample of date values
            if non_na_count > 0:
                sample = judges_df[col].dropna().iloc[0:3].tolist()
                print(f"  Sample values: {sample}")
        
        print("CSV loading test complete.\n")
        return judges_df
    
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def test_build_timeline():
    """Test building the seat timeline."""
    print("Testing seat timeline construction...")
    
    try:
        service_df = load_fjc_csv('federal-judicial-service.csv')
        print(f"Loaded federal-judicial-service.csv with {len(service_df)} rows")
        
        timeline_df = build_seat_timeline(service_df)
        print(f"Built seat timeline with {len(timeline_df)} rows")
        
        # Check for unique seat IDs
        unique_seats = timeline_df['seat_id'].nunique()
        print(f"Number of unique seat IDs: {unique_seats}")
        
        # Check for fixes of termination dates
        fix_count = 0
        for seat_id in timeline_df['seat_id'].unique():
            seat_data = timeline_df[timeline_df['seat_id'] == seat_id].sort_values('commission_date')
            if len(seat_data) <= 1:
                continue
                
            for i in range(len(seat_data) - 1):
                curr_term = seat_data.iloc[i]['termination_date']
                next_comm = seat_data.iloc[i+1]['commission_date']
                
                if pd.notna(curr_term) and pd.notna(next_comm) and curr_term == next_comm:
                    fix_count += 1
        
        print(f"Fixed {fix_count} termination dates > successor commission")
        
        # Sample of the timeline
        print("\nSample of timeline (first 3 rows):")
        print(timeline_df.head(3))
        
        print("Timeline construction test complete.\n")
        return timeline_df
    
    except Exception as e:
        print(f"Error building timeline: {e}")
        return None


def test_predecessor_info():
    """Test getting predecessor information."""
    print("Testing predecessor info extraction...")
    
    try:
        timeline_df = test_build_timeline()
        if timeline_df is None:
            return None
            
        predecessor_df = get_predecessor_info(timeline_df)
        print(f"Built predecessor lookup with {len(predecessor_df)} rows")
        
        # Check for unique vacancy dates
        unique_dates = predecessor_df['vacancy_date'].nunique()
        print(f"Number of unique vacancy dates: {unique_dates}")
        
        # Sample of the lookup
        print("\nSample of predecessor lookup (first 3 rows):")
        print(predecessor_df.head(3))
        
        print("Predecessor info test complete.\n")
        return predecessor_df
    
    except Exception as e:
        print(f"Error extracting predecessor info: {e}")
        return None


def main():
    """Run all tests."""
    print("FJC Data Processing Tests\n" + "="*25)
    
    # Ensure processed directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Run tests
    test_date_parsing()
    judges_df = test_load_csv()
    timeline_df = test_build_timeline()
    predecessor_df = test_predecessor_info()
    
    # Save outputs if tests succeeded
    if timeline_df is not None:
        output_path = PROCESSED_DATA_DIR / "seat_timeline.csv"
        timeline_df.to_csv(output_path, index=False)
        print(f"Saved seat timeline to {output_path}")
        
    if predecessor_df is not None:
        output_path = PROCESSED_DATA_DIR / "seat_predecessor_lookup.csv"
        predecessor_df.to_csv(output_path, index=False)
        print(f"Saved predecessor lookup to {output_path}")
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
