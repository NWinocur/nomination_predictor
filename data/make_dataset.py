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
