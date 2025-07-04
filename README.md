# Judicial Nomination Predictor

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A project created for presentation at the end of 4geeks' Data Science and Machine Learning cohort Miami-ds-10

This project does the following:

1. Scrapes the US Courts website for pages (HTML and PDFs) containing information about judicial vacancies.  The pages are specified via `config.py`
2. Parses those year-level pages to look for links to month-level pages.
3. Follows those links to the month-level pages to look for their links to HTML tables and/or PDFs of Judicial Vacancy data, Judicial Emergency data, and Judicial Confirmation data.
4. Downloads those pages in their original file format to `data/external`
5. Normalizes malformed HTML tables (such as from January 2001) into more-readable tables.
6. Restructures pertinent HTML tables into dataframes.  This is the primary responsibility of `dataset.py`
7. Stores that information in a raw data folder.  By default, `config.py` specifies that these CSV files are stored in `data/raw`.  Despite the name "CSV" the data uses a different character as a delimiter because characters such as `,` and `.` and `-` and `/` are ubiquitous throughout the original source's fields.
8. Reads in those raw data files to build a pandas dataframe or dataframes to use as input for data cleaning.
9. Cleans the data to uniquely identify vacancy incidents and tidy away duplicates.  This is the primary responsibility of `data_cleaning.py`
10. Feature-engineers data not existing in the raw data, but which is necessary for training a machine learning model.  This is the primary responsibility of `features.py`
11. Trains a machine learning model to predict the likely time estimate until a nomination occurs or a nomination is confirmed for a judicial vacancy.  This is the primary responsibility of `modeling.py`
12. Builds a Streamlit webapp with which users can obtain estimates given their inputs (e.g. the type of vacancy, the court, the circuit, etc.) for an existing or hypothetical vacancy.  This webapp makes calls to `predict.py` which performs inference using the trained model.

## Limitations

Judicial vacancy data is available form the US Courts website for years 1981 through present, but formatting of how that data is presented has not been consistent over the years.
The initial phase of this project is only attempting to train a model using data from 2009 July and newer for two reasons:

- Data from 2009 July and newer has been hosted in a more consistent file format (all HTML instead of a mix of HTML and PDFs), simplifying web scraping and processing.
- The person who initially gave me the idea for this project informed me that she and/or her predecessors had already performed a related project using data prior to the two most recent presidents, making newer data more interesting than older data.

A future expansion of this project can expand on the web scraping capabilities to gather prior data, including PDFs dating back to the 1980s, and train a model using that more complete dataset.

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

`make test`

## Set up Python environment

`make create_environment
make requirements`

## Run ruff (code formatter) for linting or reformatting

`make lint
make format`

## Retrieving data

FIXME: this section of the readme may wind up being largely unnecessary as of de-scoping the project to only include data from 2009 July and newer.  If we end up not needing GCP-based ways of parsing data, simplify this section.

Retrieving data can be thought of as happening in two phases: scraping it from the government's site (the easy part), and then parsing it with Google Cloud Document AI (the hard part -- but easier than trying to make sense of the sometimes-messy downloads via previous, non-AI-based methods).  That means the current version of this project is, for at least the data-retrieval portion, entirely reliant upon Google Cloud.

### Google Cloud Authentication

To authenticate with Google Cloud Document AI, you will need a service account key file -- basically a more modern alternative to an API key.
To avoid having to store this file in the repository (for security), this project utilizes the environment variable `GOOGLE_APPLICATION_CREDENTIALS` pointing to the path of your service account key file.
To edit the path of this environment variable, edit it in `.env` in the project root folder (which is gitignored by default).
Remember that python's usage of `.env` files doesn't always handle the `~` shortcut symbol to a homedir properly, so you may want to use the full path to your key file.

#### Example .env file

`
GOOGLE_APPLICATION_CREDENTIALS=/home/username/.gcp/keys/path-to-your-key-file.json
GCP_PROJECT_ID="your-gcp-project-id-here"
GCP_PROCESSOR_ID="your-document-ai-processor-id-here"
`

#### Google Cloud Cli installation (not _strictly_ required, but allows checking authentication before trying to run this project)

See <https://cloud.google.com/sdk/docs/install>for how to install the Google Cloud CLI.

#### Checking authentication via Google Cloud CLI

The following command will use that to check whether your .env file correctly specifies a path to a keyfile, and whether that keyfile authenticates successfully:

`make check_auth`

### After authentication is handled

The following command will download the raw data to the `data/raw` folder:

`make data`

If you lack authentication to Google's Document AI service, this _will_ eventually fail.  It will get as far as bulk-downloading HTML files and PDFs from the USCourts.gov website, but lacking Google Cloud auth means you won't have Google's Document AI's help parsing any of the malformed or inconsistently-formed tables and tags, so you won't get usable dataframes out of them.

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
