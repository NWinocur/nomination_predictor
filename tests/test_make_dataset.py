import os

from data import make_dataset
import pandas as pd
import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

def load_fixture(file_name):
    with open(os.path.join(FIXTURE_DIR, file_name), "r", encoding="utf-8") as f:
        return f.read()

def test_generate_urls(monkeypatch):
    # Use a known local HTML fixture
    with open("tests/fixtures/pages/archive_2025.html", encoding="utf-8") as f:
        html = f.read()

    # Patch requests.get to return the local fixture instead of live HTTP
    class MockResponse:
        text = html
    monkeypatch.setattr("requests.get", lambda url: MockResponse())

    # Call function under test
    urls = make_dataset.generate_or_fetch_archive_urls()

    # Assert structure and sample content
    assert isinstance(urls, list)
    assert all("2025" in url for url in urls)
    assert any("vacancies" in url for url in urls)
    assert any("emergencies" in url for url in urls)
    assert any("confirmations" in url for url in urls)


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


