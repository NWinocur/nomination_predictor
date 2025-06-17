import os
from dotenv import load_dotenv, find_dotenv

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)

database_url = os.environ.get("DATABASE_URL")
other_variable = os.environ.get("OTHER_VARIABLE")

def generate_or_fetch_archive_urls():
    return ["https://www.uscourts.gov/fake-url-for-test"]

def fetch_html(url):
    return "<html></html>"

def extract_vacancy_table(html):
    return [{"Seat": "1", "Court": "9th Circuit"}]

def records_to_dataframe(records):
    import pandas as pd
    return pd.DataFrame(records)

def save_to_csv(df, path):
    df.to_csv(path, index=False)


def main():
    # Step 1: Determine which pages to scrape (static list or dynamic scraping)
    urls = generate_or_fetch_archive_urls()

    all_records = []

    for url in urls:
        # Step 2: Fetch and parse HTML from the URL
        html = fetch_html(url)

        # Step 3: Extract and clean tables from HTML
        records = extract_vacancy_table(html)

        # Step 4: Append to master list
        all_records.extend(records)

    # Step 5: Convert to DataFrame
    df = records_to_dataframe(all_records)

    # Step 6: Save to CSV
    save_to_csv(df, "data/raw/judicial_vacancies.csv")
