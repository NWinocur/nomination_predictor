"""
Download fixture pages for judicial vacancies from the U.S. Courts website.

This script downloads and saves HTML and PDF pages containing judicial vacancy data
from the U.S. Courts website. It organizes the downloaded files in a directory
structure by year and month for use in testing.
"""

import os

from bs4 import BeautifulSoup
import requests

# Directory to store downloaded fixture pages
FIXTURE_DIR = "tests/fixtures/pages"
# Base domain for constructing URLs
BASE_DOMAIN = "https://www.uscourts.gov"
# Base URL for the judicial vacancies archive
ARCHIVE_BASE_URL = (
    BASE_DOMAIN + "/data-news/judicial-vacancies/archive-judicial-vacancies?year="
)

# Create the base fixture directory if it doesn't exist
os.makedirs(FIXTURE_DIR, exist_ok=True)

def save_file(path: str, content: str | bytes, binary: bool = False) -> None:
    """Save content to a file, creating parent directories if needed.
    
    Args:
        path: Path where the file should be saved
        content: Content to be written to the file (str or bytes)
        binary: If True, write in binary mode. If False, write as text (UTF-8).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = 'wb' if binary else 'w'
    encoding = None if binary else 'utf-8'
    with open(path, mode, encoding=encoding) as f:
        f.write(content)

def download_year_index(year: int) -> str | None:
    """Download and save the main archive page for a specific year.
    
    Args:
        year: The year to download archive data for
        
    Returns:
        The HTML content of the downloaded page if successful, None otherwise
    """
    url = ARCHIVE_BASE_URL + str(year)
    out_path = os.path.join(FIXTURE_DIR, f"archive_{year}.html")
    try:
        print(f"→ Downloading archive page for {year}")
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        save_file(out_path, res.text)
        return res.text
    except Exception as e:
        print(f"✖ Failed to fetch year {year}: {e}")
        return None

def download_child_pages(year: int, html: str) -> None:
    """Download and save all child pages linked from a year's archive page.
    
    Handles both HTML and PDF files, saving them with appropriate extensions.
    
    Args:
        year: The year being processed (used for logging)
        html: The HTML content of the year's archive page
    """
    soup = BeautifulSoup(html, "html.parser")
    links = soup.select("a[href*='/judges-judgeships/judicial-vacancies/archive-judicial-vacancies/']")
    print(f"  ↪ Found {len(links)} child pages for {year}")

    for a in links:
        href = a["href"]
        full_url = BASE_DOMAIN + href
        parts = href.strip("/").split("/")
        try:
            y, m, report = parts[3], parts[4], parts[5]
        except Exception:
            print(f"    ✖ Skipping malformed link: {href}")
            continue

        try:
            print(f"    ↓ {report} for {y}-{m}")
            r = requests.get(full_url, timeout=10, allow_redirects=True)
            r.raise_for_status()
            
            # Determine file extension based on content type
            content_type = r.headers.get('content-type', '').lower()
            is_pdf = 'pdf' in content_type
            extension = 'pdf' if is_pdf else 'html'
            
            # Save file with appropriate extension and content
            local_path = os.path.join(FIXTURE_DIR, y, m, f"{report}.{extension}")
            save_file(local_path, r.content if is_pdf else r.text, binary=is_pdf)
            
        except Exception as e:
            print(f"    ✖ Failed to fetch {full_url}: {e}")

# Main execution block
if __name__ == "__main__":
    # Loop through all relevant years and download their data
    for year in range(1981, 2026):
        html = download_year_index(year)
        if html:
            download_child_pages(year, html)
