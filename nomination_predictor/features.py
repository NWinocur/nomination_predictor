"""This file contains code to create features for modeling, i.e. data which wasn't in the raw data but which is being made available for the model.
It shall never output to "raw" data.  It is meant to take raw or interim data as input and deliver interim data as output.


Examples of data feature-generation include utility functions for feature-engineering U.S. federal political timelines, or capabilities such as:

Given a vacancy is on record as having existed in a specified court, with a named incumbent judge, a known reason the vacancy occurred, and a known date the vacancy began, when a vacancy with equivalent data is found on other months, then it shall be treated as the same vacancy incident.
Given earlier records for a vacancy incident lack a nominee and nomination date, when a later record for the same vacancy incident has a nominee and nomination date, then the nominee and nomination date shall be merged onto that vacancy incident's records in interim data.
Given a vacancy is on record as having existed in a specified court, with a named incumbent judge, when a vacancy with equivalent court location and incumbent is found to have occurred with a different reason and/or vacancy date (e.g. a nomination is withdrawn and reopened), then the two vacancy incidents shall be treated as each deserving of their own unique record in interim data.
Given a nomination is on record as having occurred on a specified date, when filling in date-dependent feature data (e.g. which President performed the nomination), then the interim data shall be updated to include date-inferred data (e.g. a numeric ordinal indicator identifying that President who was in office on the date of nomination.
"""

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from nomination_predictor.config import PROCESSED_DATA_DIR

app = typer.Typer()

# ---------------------------------------------------------------------------
# 1.  PRESIDENTIAL TERMS  (simple flat list; None = still in office)
#    tuple = (president_number, first_inaug_date, last_day_in_office_or_None)
# ---------------------------------------------------------------------------

PRESIDENT_TERMS = [
    # Modern era (you can extend backward easily)
    (46, date(2021, 1, 20), None),            # Joseph R. Biden
    (45, date(2017, 1, 20), date(2021, 1, 20)),
    (44, date(2009, 1, 20), date(2017, 1, 20)),
    (43, date(2001, 1, 20), date(2009, 1, 20)),
    (42, date(1993, 1, 20), date(2001, 1, 20)),
    (41, date(1989, 1, 20), date(1993, 1, 20)),
    (40, date(1981, 1, 20), date(1989, 1, 20)),
    (39, date(1977, 1, 20), date(1981, 1, 20)),
    (38, date(1974, 8,  9), date(1977, 1, 20)),
    (37, date(1969, 1, 20), date(1974, 8,  9)),
    (36, date(1963, 11,22), date(1969, 1, 20)),
    (35, date(1961, 1, 20), date(1963, 11,22)),
    (34, date(1953, 1, 20), date(1961, 1, 20)),
    (33, date(1945, 4, 12), date(1953, 1, 20)),
    (32, date(1933, 3,  4), date(1945, 4, 12)),
    (31, date(1929, 3,  4), date(1933, 3,  4)),
    (30, date(1923, 8,  2), date(1929, 3,  4)),
    (29, date(1921, 3,  4), date(1923, 8,  2)),
    (28, date(1913, 3,  4), date(1921, 3,  4)),
    (27, date(1909, 3,  4), date(1913, 3,  4)),
    (26, date(1901, 9, 14), date(1909, 3,  4)),
    #  … extend farther back as needed …
]

# Pre-sort newest-first so we can break quickly in lookups
PRESIDENT_TERMS.sort(key=lambda t: t[1], reverse=True)


# ---------------------------------------------------------------------------
# 2.  BASIC HELPERS
# ---------------------------------------------------------------------------

def _election_day(year: int) -> date:
    """
    Return the U.S. general Election Day for a given year:
    'the Tuesday following the first Monday in November'.
    """
    # First Monday in November
    first_monday = date(year, 11, 1)
    while first_monday.weekday() != 0:   # 0 = Monday
        first_monday += timedelta(days=1)
    # Tuesday following that
    return first_monday + timedelta(days=1)


@lru_cache(None)
def _president_record(d: date) -> tuple[int, date]:
    """
    Return (president_number, first_inaug_date_of_that_presidency)
    for the date supplied.  Raises ValueError if out of range.
    """
    for number, start, end in PRESIDENT_TERMS:
        if d >= start and (end is None or d < end):
            return number, start
    raise ValueError(f"Date {d} out of president table range.")


# ---------------------------------------------------------------------------
# 3.  PUBLIC API FUNCTIONS
# ---------------------------------------------------------------------------

def president_number(d: date) -> int:
    """38 for Gerald Ford, 46 for Joe Biden, …"""
    return _president_record(d)[0]


def presidential_term_index(d: date) -> int:
    """
    1 for the first four-year term of that president,
    2 for the second, 3/4 for FDR, etc.
    """
    number, first_inaug = _president_record(d)
    # Years (minus a day fudge) elapsed since first inauguration
    years_elapsed = (d - first_inaug).days // 365.2425
    return int(years_elapsed // 4) + 1


def days_into_current_term(d: date) -> int:
    """1-based: inauguration day itself returns 1."""
    number, first_inaug = _president_record(d)
    term_idx = presidential_term_index(d)
    term_start = first_inaug.replace(year=first_inaug.year + 4*(term_idx-1))
    return (d - term_start).days + 1


def days_until_next_presidential_election(d: date) -> int:
    """
    Count of days from `d` (exclusive) up to the **next** presidential-year
    Election Day (years divisible by 4). Returns 0 if `d` *is* Election Day.
    """
    year = d.year
    while year % 4 != 0 or d > _election_day(year):
        year += 1
    return (_election_day(year) - d).days


def days_until_next_midterm_election(d: date) -> int:
    """
    Days until the next even-year Election Day **that is not a presidential year**.
    """
    year = d.year
    while year % 2 != 0 or year % 4 == 0 or d > _election_day(year):
        year += 1
    return (_election_day(year) - d).days


# ----------------  CONGRESS & SESSION  -------------------------------------

CONGRESS_FIRST_YEAR = 1789   # First Congress began Mar-4-1789
CONGRESS_START_MONTH_PRE20TH = 3   # March 4 before 1935
CONGRESS_START_MONTH_20TH    = 1   # Jan   3 starting 1935

def congress_number(d: date) -> int:
    """
    118  → Jan-3-2023 to Jan-3-2025, etc.
    Formula: each Congress spans two calendar years, starting in odd years.
    """
    if d < date(1789, 3, 4):
        raise ValueError("Congress did not yet exist.")
    start_year = d.year if d.year % 2 else d.year - 1
    return (start_year - CONGRESS_FIRST_YEAR) // 2 + 1


def congress_session(d: date) -> int:
    """
    1  → first (odd-year) session
    2  → second (even-year) session
    3  → anything else (special/emergency)
    """
    cong_start_year = d.year if d.year % 2 else d.year - 1
    if d.year == cong_start_year:
        return 1
    elif d.year == cong_start_year + 1:
        return 2
    else:
        return 3
    
    
def self_test():
    """Output logging statements describing today's date as a sort of minimal self-test"""
    today = date.today()
    logger.info("Today:", today)
    logger.info("President #: ", president_number(today))
    logger.info("Term index : ", presidential_term_index(today))
    logger.info("Days into term:", days_into_current_term(today))
    logger.info("Days until next midterm :", days_until_next_midterm_election(today))
    logger.info("Days until next pres.  :", days_until_next_presidential_election(today))
    logger.info("Congress # :", congress_number(today))
    logger.info("Session    :", congress_session(today))


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
