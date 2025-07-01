import os
import requests
from bs4 import BeautifulSoup

FIXTURE_DIR = "tests/fixtures/pages"
BASE_DOMAIN = "https://www.uscourts.gov"
ARCHIVE_BASE_URL = BASE_DOMAIN + "/data-news/judicial-vacancies/archive-judicial-vacancies?year="

os.makedirs(FIXTURE_DIR, exist_ok=True)

def save_file(path, content, binary=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = 'wb' if binary else 'w'
    encoding = None if binary else 'utf-8'
    with open(path, mode, encoding=encoding) as f:
        f.write(content)

def download_year_index(year):
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

def download_child_pages(year, html):
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

# Loop through all relevant years
for year in range(1981, 2026):
    html = download_year_index(year)
    if html:
        download_child_pages(year, html)
