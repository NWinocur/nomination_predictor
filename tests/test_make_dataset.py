import os

from data import make_dataset
import pandas as pd
import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

def load_fixture(file_name):
    with open(os.path.join(FIXTURE_DIR, file_name), "r", encoding="utf-8") as f:
        return f.read()

def test_generate_urls_format_and_structure():
    urls = make_dataset.generate_or_fetch_archive_urls()

    # 1. Type and non-empty
    assert isinstance(urls, list), "Function did not return a list"
    assert len(urls) > 0, "No URLs returned"

    for url in urls:
        # 2. Domain is as expected
        assert url.startswith("https://www.uscourts.gov"), f"Bad base URL: {url}"

        # 3. Likely structural rules (heuristic, revise as needed)
        assert "judicial-vacancies" in url, f"Unexpected URL pattern: {url}"
        assert url.endswith(".html") or "/archive-" in url or url[-4:].isdigit(), f"Nonstandard URL suffix: {url}"

    # 4. Spot check: Include one known URL (if fixed URL exists)
    assert any("2023" in u or "2024" in u for u in urls), "Expected year not found in URLs"

@pytest.mark.parametrize("year", [2025, 1981])
def test_generate_urls_varied_years(monkeypatch, year):
    html_fixture = load_fixture(f"archive_{year}.html")

    def mock_get(url):
        class MockResponse:
            text = html_fixture
            status_code = 200
            def raise_for_status(self): pass
        return MockResponse()

    monkeypatch.setattr("requests.get", mock_get)
    urls = make_dataset.generate_or_fetch_archive_urls(year=year)

    if year == 2025:
        assert any("2025/06/emergencies" in u for u in urls), "Expected June link missing"
        assert all(u.startswith("https://www.uscourts.gov") for u in urls)
        assert all(any(key in u for key in ["vacancies", "confirmations", "summary", "future", "emergencies"]) for u in urls)
    elif year == 1981:
        assert urls == [], "Expected empty list due to missing reports"



def test_fetch_html_returns_html(monkeypatch):
    url = "https://example.com"

    def mock_get(url):
        class MockResponse:
            text = "<html><head></head><body>Hello</body></html>"
            status_code = 200
            def raise_for_status(self): pass
        return MockResponse()

    monkeypatch.setattr("requests.get", mock_get)
    html = make_dataset.fetch_html(url)
    assert isinstance(html, str)
    assert "<body>" in html

def test_extract_table_structure():
    html = load_fixture("example_page.html")
    records = make_dataset.extract_vacancy_table(html)

    assert isinstance(records, list)
    assert len(records) > 0
    assert isinstance(records[0], dict)
    assert "Seat" in records[0], "Missing 'Seat' column"
    assert "Court" in records[0], "Missing 'Court' column"
    assert records[0]["Seat"].isdigit(), "Seat should be numeric"


def test_dataframe_structure():
    records = [{"Seat": "1", "Court": "9th Circuit"}, {"Seat": "2", "Court": "DC Circuit"}]
    df = make_dataset.records_to_dataframe(records)

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"Seat", "Court"}
    assert df.shape == (2, 2)
    assert df["Seat"].dtype == object or df["Seat"].dtype.name.startswith("int")


def test_save_and_reload_csv(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    out_file = tmp_path / "test.csv"

    make_dataset.save_to_csv(df, out_file)
    assert out_file.exists()

    df_reloaded = pd.read_csv(out_file)
    pd.testing.assert_frame_equal(df, df_reloaded)


