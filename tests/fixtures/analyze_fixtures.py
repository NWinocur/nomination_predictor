"""Analyze HTML fixtures and output structured data for analysis."""

from collections import defaultdict
import json
import os
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

START_YEAR = 1981
END_YEAR = 2025


def analyze_file(filepath):
    """Analyze an HTML file and return structured data."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")
        result = {
            "filepath": str(filepath),
            "title": soup.title.string if soup.title else None,
            "tables": [],
            "vacancy_mentions": [],
            "common_terms": defaultdict(list),
        }

        # Analyze tables
        for table in soup.find_all("table"):
            table_data = {"headers": [], "row_count": 0, "sample_rows": [], "structure": {}}

            # Get headers
            headers = []
            thead = table.find("thead")
            if thead:
                header_row = thead.find("tr")
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

            if not headers:
                first_row = table.find("tr")
                if first_row:
                    headers = [th.get_text(strip=True) for th in first_row.find_all(["th", "td"])]

            table_data["headers"] = headers

            # Get row data
            rows = table.find_all("tr")
            table_data["row_count"] = len(rows)

            # Sample first few rows of data
            sample_size = min(3, len(rows))
            for row in rows[:sample_size]:
                cells = row.find_all(["td", "th"])
                if cells:  # Only add non-empty rows
                    table_data["sample_rows"].append([cell.get_text(strip=True) for cell in cells])

            result["tables"].append(table_data)

        # Look for common terms
        common_terms = {
            "vacancy": "vacanc",
            "court": "court",
            "judge": "judge",
            "nominee": "nominee",
            "date": "date",
            "status": "status",
            "hearing": "hearing",
            "confirmation": "confirm",
            "senate": "senate",
            "president": "president",
        }

        for term, pattern in common_terms.items():
            elements = soup.find_all(string=lambda text: pattern in str(text).lower())
            unique_texts = set(str(el).strip() for el in elements if len(str(el).strip()) > 5)
            if unique_texts:
                result["common_terms"][term] = list(unique_texts)[:10]  # Limit to 10 examples

        # Look for vacancy-related text
        vacancy_terms = ["vacanc", "nominee", "judge", "seat", "court", "hearing", "confirmation"]
        for text in soup.stripped_strings:
            if any(term in text.lower() for term in vacancy_terms):
                if 10 < len(text) < 200:  # Reasonable length
                    result["vacancy_mentions"].append(text)
                    if len(result["vacancy_mentions"]) >= 20:  # Limit to 20 mentions
                        break

        return result

    except Exception as e:
        return {"filepath": str(filepath), "error": str(e)}


def analyze_fixtures(output_file="fixture_analysis.json"):
    """Analyze all HTML fixtures and save results to a JSON file."""
    fixture_dir = Path(__file__).parent / "fixtures" / "pages"
    results = {
        "archive_pages": [],
        "monthly_reports": [],
        "summary": {
            "total_files": 0,
            "tables_found": 0,
            "common_terms": defaultdict(int),
            "file_errors": 0,
        },
    }

    # Track terms across all files
    all_terms = defaultdict(list)

    def process_file(filepath, category):
        print(f"Analyzing: {filepath}")
        analysis = analyze_file(filepath)

        if "error" in analysis:
            results["summary"]["file_errors"] += 1
            return

        results["summary"]["total_files"] += 1

        # Add to appropriate category
        if category == "archive":
            results["archive_pages"].append(analysis)
        else:
            results["monthly_reports"].append(analysis)

        # Update summary stats
        results["summary"]["tables_found"] += len(analysis.get("tables", []))

        # Track common terms
        for term, examples in analysis.get("common_terms", {}).items():
            all_terms[term].extend(examples)

    print("\nProcessing archive pages...")
    for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Years"):
        archive_file = fixture_dir / f"archive_{year}.html"
        if archive_file.exists():
            process_file(archive_file, "archive")
        else:
            print(f"  Archive file not found for year {year}")

    # Process monthly reports (all years and months)
    print("\nProcessing monthly reports...")
    year_dirs = sorted(
        (
            d
            for d in fixture_dir.glob("*")
            if d.is_dir() and d.name.isdigit() and START_YEAR <= int(d.name) <= END_YEAR
        ),
        reverse=True,
    )

    # Find all monthly report files first
    monthly_reports = []
    for year_dir in year_dirs:
        for month_dir in sorted(year_dir.glob("*"), reverse=True):
            if not month_dir.is_dir() or not month_dir.name.isdigit():
                continue
            for report in month_dir.glob("*.html"):
                if "vacancies" in report.name.lower():
                    monthly_reports.append(report)
    
    # Process all found reports with progress bar
    for report in tqdm(monthly_reports, desc="Monthly reports"):
        process_file(report, "monthly")

    # Add term frequency to summary
    for term, examples in all_terms.items():
        results["summary"]["common_terms"][term] = len(examples)

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis complete. Results saved to {output_file}")
    print(f"Total files analyzed: {results['summary']['total_files']}")
    print(f"Tables found: {results['summary']['tables_found']}")
    print(f"Common terms: {dict(results['summary']['common_terms'])}")


if __name__ == "__main__":
    analyze_fixtures()
