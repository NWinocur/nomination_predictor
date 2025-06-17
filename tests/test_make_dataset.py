import os

from data import make_dataset
import pandas as pd
import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

def load_fixture(file_name):
    with open(os.path.join(FIXTURE_DIR, file_name), "r", encoding="utf-8") as f:
        return f.read()

def test_generate_urls():
    urls = make_dataset.generate_or_fetch_archive_urls()
    assert isinstance(urls, list)
    assert all(url.startswith("https://www.uscourts.gov") for url in urls)
    assert len(urls) > 0

def test_fetch_html(monkeypatch):
    url = "https://example.com"

    def mock_get(url):
        class MockResponse:
            text = "<html></html>"
        return MockResponse()

    monkeypatch.setattr("requests.get", mock_get)
    html = make_dataset.fetch_html(url)
    assert isinstance(html, str)
    assert "<html>" in html

def test_extract_vacancy_table():
    html = load_fixture("example_page.html")
    records = make_dataset.extract_vacancy_table(html)
    assert isinstance(records, list)
    assert isinstance(records[0], dict)

def test_records_to_dataframe():
    records = [{"Seat": "1", "Court": "9th Circuit"}, {"Seat": "2", "Court": "DC Circuit"}]
    df = make_dataset.records_to_dataframe(records)
    assert isinstance(df, pd.DataFrame)
    assert "Court" in df.columns

def test_save_to_csv(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    out_file = tmp_path / "test.csv"
    make_dataset.save_to_csv(df, out_file)
    assert out_file.exists()
    reloaded = pd.read_csv(out_file)
    pd.testing.assert_frame_equal(df, reloaded)
