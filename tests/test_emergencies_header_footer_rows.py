"""Tests for ensuring emergencies scraper correctly skips non-data footer rows."""

from pathlib import Path

from nomination_predictor.emergencies_scraper import extract_emergencies_table


def get_pre_downloaded_emergencies_html_from(year: int, month_num: str) -> str:
    """Return the content of a real emergencies page from the fixtures."""
    path = Path(__file__).parent / "fixtures" / "pages" / str(year) / month_num / "emergencies.html"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def test_legacy_emergencies_skip_footer_rows():
    """Test that non-data footer rows in legacy tables are not included in results."""
    # Get the HTML content from a legacy emergencies file
    html = get_pre_downloaded_emergencies_html_from(2010, "01")
    
    # Extract the data
    records = extract_emergencies_table(html)
    
    # Verify we have the expected number of records (33 emergencies in Jan 2010)
    assert len(records) == 33, f"Expected 33 emergency records, got {len(records)}"
    
    # Verify no record contains footer-like content
    footer_phrases = [
        'Total Emergencies',
        'judicial emergency is defined',
        'data is current as of'
    ]
    
    for record in records:
        for field, value in record.items():
            if isinstance(value, str):
                for phrase in footer_phrases:
                    assert phrase.lower() not in value.lower(), \
                        f"Found footer content in record field {field}: {value}"
