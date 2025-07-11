"""
Federal Judicial Center (FJC) Data Processor Module

This module provides functionality for processing and validating FJC data files.
It includes both programmatic interfaces and command-line functionality.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
import pandas as pd

from nomination_predictor.config import configure_logging
from nomination_predictor.fjc_data import (
    FJC_DATA_DIR,
    build_seat_timeline,
    get_predecessor_info,
    load_fjc_csv,
    load_judges_data,
    parse_fjc_date,
)


def validate_data_files() -> bool:
    """
    Check that all required FJC data files exist.
    
    Returns:
        bool: True if all required files exist, False otherwise
    """
    required_files = ['federal-judicial-service.csv', 'judges.csv']
    optional_files = ['demographics.csv', 'nominations-unsuccessful.csv']
    missing = []
    
    for file in required_files:
        if not (FJC_DATA_DIR / file).exists():
            missing.append(file)
    
    if missing:
        logger.error(f"Missing required FJC data files: {missing}")
        logger.error(f"Expected in: {FJC_DATA_DIR}")
        return False
    
    logger.info(f"All required FJC data files found in {FJC_DATA_DIR}")
    
    # Report on optional files
    for file in optional_files:
        if (FJC_DATA_DIR / file).exists():
            logger.info(f"Optional file found: {file}")
        else:
            logger.info(f"Optional file not found: {file}")
    
    return True


def demonstrate_date_parsing() -> None:
    """
    Demonstrate date parsing functionality with various formats.
    
    NOTE: This is not a test function but a demonstration. Proper unit tests for 
    parse_fjc_date exist in tests/test_fjc_data.py.
    
    This function demonstrates how the parse_fjc_date function handles
    different date formats commonly found in FJC data.
    """
    logger.info("Demonstrating date parsing functionality...")

    test_dates_str = [
        "1889-03-15",  # Excel format
        "03/15/1889",  # CSV format
        "1/5/2000",    # Single digit month/day
        "",            # Empty
        "invalid",     # Invalid
    ]
    
    # Handle None separately to avoid type issues
    logger.info(f"'None' -> {parse_fjc_date(None)}")

    for date_str in test_dates_str:
        parsed = parse_fjc_date(date_str)
        logger.info(f"'{date_str}' -> {parsed}")

    logger.info("Date parsing demonstration complete")


def process_fjc_data(
    output_dir: Path, 
    validate_mode: bool = False
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Process FJC data files and output to the specified directory.
    
    Args:
        output_dir: Directory to save output files
        validate_mode: If True, perform data validation without saving
        
    Returns:
        Tuple containing (seat_timeline_df, judges_df, predecessor_df) or
        (None, None, None) if processing fails
    """
    if not validate_data_files():
        return None, None, None
    
    # Load service data
    try:
        logger.info("Loading federal-judicial-service.csv")
        service_df = load_fjc_csv('federal-judicial-service.csv')
        logger.info(f"Loaded {len(service_df)} service records")
        
        # Basic data validation
        if validate_mode:
            missing_seat_ids = service_df['seat_id'].isna().sum()
            if missing_seat_ids > 0:
                logger.warning(f"{missing_seat_ids} records missing seat_id ({missing_seat_ids/len(service_df):.2%})")
                
            invalid_courts = service_df['court'].isna().sum()
            if invalid_courts > 0:
                logger.warning(f"{invalid_courts} records with invalid court ({invalid_courts/len(service_df):.2%})")
    except Exception as e:
        logger.error(f"Error loading service data: {str(e)}")
        return None, None, None
        
    # Load judges data
    try:
        logger.info("Loading judges.csv")
        judges_df = load_judges_data(include_demographics=True)
        logger.info(f"Loaded {len(judges_df)} judge records")
        
        # Basic data validation
        if validate_mode:
            missing_names = (judges_df['first_name'].isna() | judges_df['last_name'].isna()).sum()
            if missing_names > 0:
                logger.warning(f"{missing_names} judge records missing name ({missing_names/len(judges_df):.2%})")
    except Exception as e:
        logger.error(f"Error loading judges data: {str(e)}")
        return None, None, None
    
    # Build seat timeline
    try:
        logger.info("Building seat timeline")
        seat_timeline_df = build_seat_timeline(service_df)
        logger.info(f"Built seat timeline with {len(seat_timeline_df)} records")
        
        # Validate the seat timeline
        if validate_mode:
            seat_ids = seat_timeline_df['seat_id'].nunique()
            logger.info(f"Found {seat_ids} unique seats in timeline")
            
            missing_commission = seat_timeline_df['commission_date'].isna().sum()
            if missing_commission > 0:
                logger.warning(f"{missing_commission} records missing commission_date ({missing_commission/len(seat_timeline_df):.2%})")
            
            # Check for succession gaps (missing predecessors)
            seat_ids_list = seat_timeline_df['seat_id'].unique()
            for seat_id in seat_ids_list[:min(10, len(seat_ids_list))]:  # Check first 10 seats as sample
                seat_rows = seat_timeline_df[seat_timeline_df['seat_id'] == seat_id].sort_values('commission_date')
                if len(seat_rows) <= 1:
                    continue
                
                for i in range(1, len(seat_rows)):
                    prev_term_date = seat_rows.iloc[i-1]['termination_date']
                    curr_comm_date = seat_rows.iloc[i]['commission_date']
                    if pd.notna(prev_term_date) and pd.notna(curr_comm_date):
                        gap_days = (curr_comm_date - prev_term_date).days
                        if gap_days < -30:  # Overlap of more than 30 days
                            logger.warning(f"Potential overlap in seat {seat_id}: {gap_days} days between terms")
                        elif gap_days > 365*4:  # Gap of more than 4 years
                            logger.warning(f"Potential gap in seat {seat_id}: {gap_days} days between terms")
    except Exception as e:
        logger.error(f"Error building seat timeline: {str(e)}")
        return None, None, None
        
    # Create predecessor lookup
    try:
        logger.info("Creating predecessor lookup")
        predecessor_df = get_predecessor_info(seat_timeline_df)
        logger.info(f"Created predecessor lookup with {len(predecessor_df)} records")
    except Exception as e:
        logger.error(f"Error creating predecessor lookup: {str(e)}")
        return None, None, None
    
    # Save processed data if not in validation mode
    if not validate_mode:
        logger.info(f"Saving processed data to {output_dir}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save seat timeline
        seat_timeline_path = output_dir / "seat_timeline.csv"
        seat_timeline_df.to_csv(seat_timeline_path, index=False)
        logger.info(f"Saved seat timeline to {seat_timeline_path}")
        
        # Save predecessor lookup
        predecessor_path = output_dir / "seat_predecessors.csv"
        predecessor_df.to_csv(predecessor_path, index=False)
        logger.info(f"Saved predecessor lookup to {predecessor_path}")
        
        # Save judges data
        judges_path = output_dir / "judges.csv"
        judges_df.to_csv(judges_path, index=False)
        logger.info(f"Saved judges data to {judges_path}")
    
    logger.info("FJC data processing complete")
    return seat_timeline_df, judges_df, predecessor_df


def run_tests() -> None:
    """
    Run tests for the FJC data processing functionality.
    
    This function demonstrates and tests various aspects of the FJC data
    processing pipeline using actual data files.
    """
    logger.info("Running FJC data processing tests")
    
    # Demonstrate date parsing (not a test)
    demonstrate_date_parsing()
    
    # Test data file validation
    valid_files = validate_data_files()
    if not valid_files:
        logger.warning("Skipping remaining tests due to missing data files")
        return
    
    # Test processing with validation mode
    logger.info("Testing data processing in validation mode")
    seat_timeline_df, judges_df, predecessor_df = process_fjc_data(
        output_dir=FJC_DATA_DIR,
        validate_mode=True
    )
    
    if seat_timeline_df is not None:
        logger.info(f"Seat timeline test succeeded with {len(seat_timeline_df)} records")
    else:
        logger.error("Seat timeline test failed")
    
    logger.info("All tests completed")


def main() -> None:
    """Main function to parse arguments and run processing."""
    parser = argparse.ArgumentParser(description="Process FJC data files")
    parser.add_argument("--validate", action="store_true", help="Run in validation mode")
    parser.add_argument("--test", action="store_true", help="Run tests instead of processing")
    parser.add_argument("--output", default=str(FJC_DATA_DIR), 
                        help=f"Output directory (default: {FJC_DATA_DIR})")
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level="INFO")
    
    if args.test:
        run_tests()
        return
        
    output_dir = Path(args.output)
    logger.info(f"Output directory: {output_dir}")
    
    if args.validate:
        logger.info("Running in validation mode")
        
    # Process the data
    process_fjc_data(output_dir, validate_mode=args.validate)


if __name__ == "__main__":
    main()
