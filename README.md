# Judicial Nomination Predictor

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A project created for presentation at the end of 4geeks' Data Science and Machine Learning cohort Miami-ds-10

This project does the following:

1. Scrapes the US Courts website for pages (HTML and PDFs) containing information about judicial vacancies.  The pages are specified via `config.py`
2. Extracts information from those pages by parsing year-level pages to look for links to month-level pages, then following those links to the month-level pages to look for their links to HTML tables and/or PDFs of vacancy data.  Web scraping, following links to HTML pages and PDFs, and parsing HTML tables and PDFs is the primary responsibility of `dataset.py`
3. Stores that information in a raw data folder.  By default, `config.py` specifies that these CSV files are stored in `data/raw`.  Despite the name "CSV" the data uses a different character as a delimiter because characters such as `,` and `.` and `-` and `/` are ubiquitous throughout the original source's fields.
4. Reads in those raw data files to build a pandas dataframe or dataframes to use as input for data cleaning.
5. Cleans the data to uniquely identify vacancy incidents and tidy away duplicates.  This is the primary responsibility of `data_cleaning.py`
6. Feature-engineers data not existing in the raw data, but which is necessary for training a machine learning model.  This is the primary responsibility of `features.py`
7. Trains a machine learning model to predict the likely time estimate until a nomination occurs or a nomination is confirmed for a judicial vacancy.  This is the primary responsibility of `modeling.py`
8. Builds a Streamlit webapp with which users can obtain estimates given their inputs (e.g. the type of vacancy, the court, the circuit, etc.) for an existing or hypothetical vacancy.  This webapp makes calls to `predict.py` which performs inference using the trained model.

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
├── setup.cfg          <- Configuration file for development-related tools (e.g. autoformatter settings for Ruff, if not defaults)
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

### Running tests

make test

## Set up Python environment

make create_environment
make requirements

## Run ruff (code formatter) for linting or reformatting

make lint
make format

## Retrieving data

make data

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

### Notebook Naming Convention

This project borrows from <https://cookiecutter-data-science.drivendata.org/using-the-template/> a consistent naming pattern for Jupyter notebooks to maintain organization and clarity:

```
<PHASE>.<SEQUENCE>-<INITIALS>-<DESCRIPTION>.ipynb
```

- **PHASE**: Number indicating the development phase
  - `0` - Data exploration - often just for exploratory work
  - `1` - Data cleaning and feature creation - often writes data to data/processed or data/interim
  - `2` - Visualizations - often writes publication-ready viz to reports
  - `3` - Modeling - training machine learning models
  - `4` - Publication/Reporting - Notebooks that get turned directly into reports
- **SEQUENCE**: Two-digit number for ordering notebooks within the same phase
- **INITIALS**: Your initials (lowercase)
- **DESCRIPTION**: Short hyphen-separated description of the notebook's purpose

Example: `0.01-pjb-data-source-1.ipynb`

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future improvement opportunities

- Include handling of future vacancies.  It is plausible that a vacancy which has been announced in advance (e.g. a retirement) could be filled more quickly than one which occurred due to a sudden vacancy (e.g. a death).  Handling the "future vacancy list" properly would require deciding how to handle vacancies whose vacancy date is "TBD" (there are multiple such instances on [Future Judicial Vacancies for December 2018](https://www.uscourts.gov/judges-judgeships/judicial-vacancies/future-judicial-vacancies-december-2018)), and because the one thing all other vacancies have is a date on which they occurred, this is a non-trivial question.

## Acknowledgments

- Data source: [US Courts - Judicial Vacancies](https://www.uscourts.gov/judges-judgeships/judicial-vacancies)
- Project structure based on [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
