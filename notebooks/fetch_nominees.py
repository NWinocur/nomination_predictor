"""
This script demonstrates how to fetch nominee-specific data from Congress.gov API
and save it as a separate DataFrame.

Usage:
1. Make sure you have a nominations DataFrame loaded (nominations_df)
2. Run this code to fetch and save nominee-specific data
"""

import os
from pathlib import Path

import pandas as pd

from nomination_predictor.config import EXTERNAL_DATA_DIR
from nomination_predictor.congress_api import CongressAPIClient

# Define cache file path for nominees
nominees_cache_file = os.path.join(EXTERNAL_DATA_DIR, "congress_nominees_cache.csv")

def fetch_and_save_nominees(nominations_df, congress_client):
    """
    Fetch nominee data from nominations DataFrame and save to CSV.
    
    Args:
        nominations_df: DataFrame with nomination records containing nominee_url field
        congress_client: Initialized CongressAPIClient instance
        
    Returns:
        DataFrame with nominee-specific data
    """
    # Check if we have cached data
    if os.path.exists(nominees_cache_file):
        print(f"Found cached nominees data at {nominees_cache_file}")
        nominees_df = pd.read_csv(nominees_cache_file)
        print(f"Loaded {len(nominees_df)} nominee records from cache")
        return nominees_df
    
    # If no cache, fetch from API
    print(f"Fetching nominee data for {len(nominations_df)} nominations...")
    
    # Check if nominee_url column exists
    if 'nominee_url' not in nominations_df.columns:
        print("⚠️ No nominee_url column found in nominations_df")
        return pd.DataFrame()
    
    # Filter out records without nominee_url
    valid_nominations = nominations_df[~nominations_df['nominee_url'].isna()]
    print(f"Found {len(valid_nominations)} nominations with valid nominee_url")
    
    # Fetch nominee data for all nominations
    nominees_data = congress_client.get_all_nominees_data(valid_nominations)
    
    # Convert to DataFrame
    nominees_df = pd.DataFrame(nominees_data)
    print(f"\nTotal nominees retrieved: {len(nominees_df)}")
    
    # Save to CSV
    if nominees_df is not None and not nominees_df.empty:
        nominees_df.to_csv(nominees_cache_file, index=False)
        print(f"✓ Saved {len(nominees_df)} nominees to cache")
        
    return nominees_df

# Example usage in notebook:
# nominees_df = fetch_and_save_nominees(nominations_df, congress_client)
# print(nominees_df.head())

# Optional: Display column statistics
# print("\nNominees DataFrame columns:")
# for col in sorted(nominees_df.columns):
#     print(f"- {col}: {nominees_df[col].nunique()} unique values")
