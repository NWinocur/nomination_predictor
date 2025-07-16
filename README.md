# Judicial Nomination Predictor

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A project created for presentation at the end of 4geeks' Data Science and Machine Learning cohort Miami-ds-10

This project's notebooks shall perform the following:

Notebook 0 shall:

1. Call `dataset.py` to either look for and use already-downloaded Federal Court information from the Federal Judicial Center (FJC), or if no such data is found (or if the input arg to force a data refresh is specified), download it with the help of `fjc_data.py`. The URLs to the FJC excel spreadsheets and CSV files are specified via `config.py`.  Any data saved by `dataset.py` shall be saved to `data/external`.
2. Call `dataset.py` to either look for and use already-downloaded Congress information from the US Congress API, or if no such data is found (or if the input arg to force a data refresh is specified), download it using the API key specified via environment variable `CONGRESS_API_KEY`.
3. Convert the cached or downloaded data -- with as few changes to their values as feasible just yet -- into dataframe(s), which is then be saved into `data/raw`.

Notebook 1 shall:

1. Load dataframe(s) that Notebook 0 had saved into `data/raw`.
2. Call `features.py` functions to perform cleaning and feature-engineering, saving interim work to `data/interim`.

Notebook 2 shall:

1. Load dataframe(s) that Notebook 1 had saved into `data/interim`
2. Call `plots.py` to generate visualizations of the data and provide exploratory data analysis.
3. Make whatever adjustments to data appear appropriate given interpretations made during exploratory data analysis, and save the resulting dataframe(s) to `data/processed`.

Notebook 3 shall:

1. Load dataframe(s) that Notebook 2 had saved to `data/processed`
2. Call `modeling.py` to train a machine learning model (TBD as of writing this: XGBoost or Regularized Linear Regression) to predict the likely time estimate between when a vacancy starts and when a nomination occurs, or the predict the likely time estimate between when a vacancy starts and when the replacement appointee is confirmed, or predict the likely time estimate between when a person is nominated and when their nomination is confirmed.
3. Evaluate trained model or models to select which to use in subsequent steps
4. Save the trained model to `models/`.

Notebook 4 (_not yet implemented as of typing this_) shall:

1. Load the model that Notebook 3 had saved to `models/`.
2. Be a proof-of-concept space for a follow-up script to build a Streamlit webapp with which users can obtain estimates given their inputs (e.g. the type of vacancy, the court, the circuit, etc.) for an existing or hypothetical vacancy. This webapp shall make calls to `predict.py` which performs inference using the trained model.

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

```bash
make test
```

## Set up Python environment

```bash
pip install virtualenvwrapper
make create_environment
make requirements
```

## Run ruff (code formatter) for linting or reformatting

```bash
make lint
make format
```

## Retrieving data

If you don't already have your CONGRESS_API_KEY environment variable assigned via .env file, you will need to do so before running `make data`.

#### Example .env file

```.env
CONGRESS_API_KEY="your-congress-api-key-here"
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

### Notebook Naming Convention

This project borrows from <https://cookiecutter-data-science.drivendata.org/using-the-template/> a consistent naming pattern for Jupyter notebooks to maintain organization and clarity:

`
<PHASE>.<SEQUENCE>-<INITIALS>-<DESCRIPTION>.ipynb
`

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

- Data source: History of the Federal Judiciary. <http://www.fjc.gov>. Web site of the Federal Judicial Center, Washington, DC. 
- Data source: Congress.gov API <https://api.congress.gov>. Hosted by the United States Congress, Washington, DC.
- Data source (used by earlier versions of code in this same repository): [US Courts - Judicial Vacancies](https://www.uscourts.gov/judges-judgeships/judicial-vacancies)
- Project structure based on [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
