# Judicial Nomination Predictor

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A project created for presentation at the end of 4geeks' Data Science and Machine Learning cohort Miami-ds-10

This project does the following:

1. Scrapes the US Courts website for information about judicial vacancies
2. Builds a machine learning model to predict the likely time estimate until a nomination is confirmed for a judicial vacancy
3. Builds a Streamlit webapp with which users can obtain estimates given their inputs (e.g. the type of vacancy, the court, the circuit, etc.) for an existing or hypothetical vacancy.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         nomination_predictor and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── nomination_predictor   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes nomination_predictor a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Tests

## 🔌 Test Fixtures: Judicial Vacancy Pages

Some of this project's test cases uses locally-downloaded copies of real HTML pages from [uscourts.gov](https://www.uscourts.gov/data-news/judicial-vacancies/archive-judicial-vacancies) as test fixtures to validate data-scraping behavior.

### 🧪 Fixture Location
Fixture files are saved in:

```
tests/fixtures/pages
```

These are used by the test suite for deterministic, offline testing.

### 🚫 Not Version Controlled
Due to their size (~371MB total), these files are **not committed to Git**. If cloning the repository fresh and wanting to run tests, you’ll need to regenerate them.

### 📥 Regenerate Fixtures

To re-download all year/month-level HTML files:

```bash
python tests/fixtures/download_fixture_pages.py
```

This will populate tests/fixtures/pages/ with HTML documents across the full range of years.

# Run flake8
flake8 data tests

# Run black (code formatter)
black data tests

# Run isort (import sorter)
isort data tests

```

## Project Structure

```
├── .github/               # GitHub workflows and issue templates
├── data/
│   ├── external/         # Data from third party sources
│   ├── interim/          # Intermediate data that has been transformed
│   ├── processed/        # The final, canonical data sets for modeling
│   └── raw/              # The original, immutable data dump
├── docs/                 # Documentation
├── models/               # Trained and serialized models
├── notebooks/            # Jupyter notebooks for exploration
├── references/           # Data dictionaries, manuals, etc.
├── reports/              # Generated analysis and visualizations
└── nomination_predictor/ # Source code
    ├── __init__.py
    ├── config.py         # Configuration settings
    ├── data/             # Data processing code
    ├── features/         # Feature engineering
    ├── models/           # Model training and evaluation
    └── visualization/    # Visualization code
```

## Data Dictionary

See [references/data_dictionary.md](references/data_dictionary.md) for a detailed description of the data fields.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Data source: [US Courts - Judicial Vacancies](https://www.uscourts.gov/judges-judgeships/judicial-vacancies)
- Project structure based on [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
