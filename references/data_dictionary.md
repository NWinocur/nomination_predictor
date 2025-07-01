# Judicial Vacancies Data Dictionary

This document describes the structure and content of the judicial vacancies dataset.

## Overview

The dataset contains information about current and historical judicial vacancies in the United States federal court system. This data is meant to be collected from public information sources such as the [US Courts website](https://www.uscourts.gov/judges-judgeships/judicial-vacancies).

Field names were chosen for a balance between intuitive clarity vs. backwards-compatibility with the data dictionary from a preceding project by Wendy L. Martinek of Binghamton University et. al. (see "Lower Federal Court Confirmation Database February-March 2005 Release").

## Prediction Fields

These are the primary fields this project's problem statement seeks to be able to predict for still-pending and future vacancies.

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|----------|
| `nomination_days` | Integer | Number of days from vacancy to nomination | 54 |
| `confirmation_days` | Integer | Number of days from nomination to confirmation | 71 |

## Data Fields

### Core Fields

These are fields directly obtainable from the US Courts website.

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|----------|
| `vacancy_id` | Integer | Unique identifier for the judicial vacancy incident | 1 |
| `district` | String | District (can include both circuit number and state-region signifiers) | (see separate table below) |
| `incumbent` | String | Name of the incumbent; raw data formats names as "Lastname,Firstname" or "Lastname,Firstname M." | "Smith,John A." |
| `vacancy_reason` | Integer | Reason nomination opportunity arises, coded among one of reasons specified in separate table below | 2 |
| `vacancy_notification_date` | Date | Applicable for Future Judicial Vacancies; date on which notification was delivered that a vacancy will occur | 2025-01-15 |
| `vacancy_date` | Date | Date on which seat originally became (or in the case of Future Vacancies, will become) open | 2025-01-15 |
| `nominee` | String | Name of the nominee; empty string if nobody yet nominated; None if unknown. | "Burdell,James P." |
| `nomination_date` | Date | Date when the nomination was made by the then-acting president | 2025-03-10 |
| `confirmation_date` | Date | Date when the nominee was confirmed (if applicable) | 2025-05-20 |
| `vacancy_days_pending` | Integer | Number of days the position has been vacant | 125 |
| `weighted_filings_per_judgeship` | Integer | Weighted Filings per Judgeship | 542 |
| `adjusted_filings_per_panel` | Integer | Adjusted Filings per Panel | 0 |

#### District examples

| Raw | Description |
|-----|-------------|
| 09 - MT | 9th circuit Montana |
| 11 - FL-M | 11th circuit Middle district of Florida |
| CL | federal Claims court |
| 06 - CCA | 6th Circuit Court of Appeals |
| DC - DC | United States District Court for the District of Columbia |

#### Reason nomination opportunity arises

| Code | Raw | Description |
|------|-----|-------------|
| 1 | Retired | Incumbent retired |
| 2 | Senior | Incumbent took senior status |
| 3 | Elevated | Incumbent elevated to another position |
| 4 | Deceased | Incumbent died |
| 5 | New Position | New seat |
| 6 | Withdrew | President withdrew previous nomination |
| 7 | Returned | Previous nomination returned to the president |
| 8 | Resigned | Incumbent resigned |
| 9 | Impeached | Incumbent impeached |
| 10 | Unfilled | New president assumes office and prior vacancy left unfilled |
| 11 | Recess | Recess appointment |

### Court Information

These are fields derivable from the "district" field.

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|----------|
| `court_type` | int | Type of lower court | 0 = district, 1 = circuit |
| `circuit` | String | Circuit number (for Circuit Courts) | "9" |
| `state` | String | State where the court is located | "California" |

### Presidential information

These are fields derivable based on other fields of type Date such as "nomination_date".

Backwards-incompatibility note: fields such as "prez_v" do NOT adopt the predecessor project's "PREZ_V" encoding exactly: e.g. where our predecessor project encoded Ford as President 1, this project encodes Ford as the US' 38th president, etc.  This is meant to allow easier recognition at-a-glance, and easier integration with longer-ago historical data, if that becomes available.

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|----------|
| `prez_v` | int | which US president was in office when seat originally became open | 38 = Ford, 39 = Carter, etc. |
| `prez_st` | int | which US president was in office when nomination opportunity began | 38 = Ford, 39 = Carter, etc. |
| `prez_n` | int | which US president was in office when nomination was made | 38 = Ford, 39 = Carter, etc. |
| `prez_conf` | int | which US president was in office when nomination was confirmed | 38 = Ford, 39 = Carter, etc. |
| `prez_final` | int | which US president was in office at time of final congressional action | 38 = Ford, 39 = Carter, etc. |
| `prez_term` | int | term of president when nomination made | 1 = during first term of president; 2 = during second term of president |
| `prez_year` | int | year of presidential term when nomination made | 1, 2, 3, or 4 |
| `prez_approval` | float | Percentage of survey respondents approving of presidential job performance based on Gallup poll closest to date of nomination | 0.33 |
| `prez_end` | int | which US president was in office when nomination opportunity ended | 38 = Ford, 39 = Carter, etc. |
| `prez_dur` | int | Number of days between the start and end of the nomination opportunity | 125 |

### Congressional information

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|----------|
| `cong_v` | int | Congressional term in which seat originally became open | 116 |
| `sess_v` | int | Congressional session in which Congressional session in which seat originally became open | 1 = first session, 2 = 2nd session, 3 = special session |
| `cong_n` | int | Congressional term in which nomination was made | 116 |
| `sess_n` | int | Congressional session in which nomination was made | 1 = first session, 2 = 2nd session, 3 = special session |
| `sess_v` | int | Congressional session in which Congressional session in which seat originally became open | 1 = first session, 2 = 2nd session, 3 = special session |
| `cong_c` | int | Congressional term in which nominee confirmed | 116 |
| `sess_c` | int | Congressional session in which nominee confirmed | 1 = first session, 2 = 2nd session, 3 = special session |
| `final_c_action_date` | Date | Date of final congressional action (Note: Final congressional action consists of confirmation, rejection, return of nomination to president (e.g., when Congress adjourns since die), or withdrawal of nomination by president) | 2018-08-28 |
| `cong_f` | int | Congressional term in which final congressional action occurs | 116 |
| `sess_f` | int | Congressional session in which final congressional action occurs | 1 = first session, 2 = 2nd session, 3 = special session |
| `c_final_action` | int | encoded indicator of final congressional action on nomination | 1 = confirmed, 2 = returned to president, 3 = withdrawn by president |

### Unified vs. Divided government

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|----------|
| `nom_is_unified` | Boolean | At time of nomination, President's party holds a majority in both the House and the Senate. | false |
| `nom_is_div_opp_house` | Boolean | At time of nomination, President's party holds a majority in the Senate, but the opposition controls the House. | false |
| `nom_is_div_opp_senate` | Boolean | At time of nomination, President's party holds a majority in the House, but the opposition controls the Senate. | true |
| `nom_is_fully_div` | Boolean | At time of nomination, opposition party controls both the House and the Senate. | false |
| `conf_is_unified` | Boolean | At time of confirmation, President's party holds a majority in both the House and the Senate. | false |
| `conf_is_div_opp_house` | Boolean | At time of confirmation, President's party holds a majority in the Senate, but the opposition controls the House. | false |
| `conf_is_div_opp_senate` | Boolean | At time of confirmation, President's party holds a majority in the House, but the opposition controls the Senate. | true |
| `conf_is_fully_div` | Boolean | At time of confirmation, opposition party controls both the House and the Senate. | false |

### Nominee Specific

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|----------|
| `renom` | boolean | Is this a renomination of a previous nominee for the same seat? | false |
| `partyid` | Partisan identification of nominee (or None if unknown) | 0 = Democrat, 1 = Republican, 3 = Other | 
| `gender_is_female` | boolean | Gender of nominee | true |
| `minority_racial` | encoded indicator of racial minority affiliation, or None if not known | (see table below) | 1 |
| `aba_rating` | ABA rating of nominee | (see table below) | 5 |

#### Racial minority encoding

| Code | Description |
|------|-------------|
| 1 | White |
| 2 | African American |
| 3 | Hispanic |
| 4 | Native American |
| 5 | Asian American |

#### American Bar Association Rating

| Code | Description |
|------|-------------|
| 1 | not qualified (age) |
| 2 | not qualified |
| 3 | not qualified/qualified |
| 4 | qualified/not qualified |
| 5 | qualified |
| 6 | qualified/well qualified |
| 7 | well qualified/qualified |
| 8 | well qualified |
| 9 | well qualified/extremely well qualified |
| 10 | extremely well qualified/well qualified |
| 11 | extremely well qualified |

### Unused fields

These fields were set aside to simplify this project and may or may not be worth considering in future expanded work:

#### Nomination Status

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|----------|
| `status` | String | Current status of the nomination | "Confirmed" |
| `hearing_date` | Date | Date of confirmation hearing | 2025-04-15 |
| `committee_vote_date` | Date | Date of committee vote | 2025-04-30 |
| `full_senate_vote_date` | Date | Date of full Senate vote | 2025-05-20 |
| `vote_yea` | Integer | Number of 'Yea' votes | 65 |
| `vote_nay` | Integer | Number of 'Nay' votes | 35 |

#### Judge Information (for filled positions)

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|----------|
| `judge_name` | String | Name of the judge who left the position | "Jane Doe" |
| `judge_senior_status` | Boolean | Whether the judge took senior status | true |
| `judge_retirement_date` | Date | Date of retirement (if applicable) | 2024-12-31 |
| `judge_death_date` | Date | Date of death (if applicable) | null |

## Data Collection Notes

- Data is collected from publicly available sources on the US Courts website
- Some fields may be null or empty for certain records
- Dates are in YYYY-MM-DD format
- All text fields are case-sensitive

## Related Resources

- [US Courts - Judicial Vacancies](https://www.uscourts.gov/judges-judgeships/judicial-vacancies)
- [US Senate - Nominations](https://www.senate.gov/legislative/nominations.htm)
- [Federal Judicial Center - Biographical Directory](https://www.fjc.gov/history/judges/search/advanced-search)

## Change Log

| Date | Description |
|------|-------------|
| 2025-06-25 | Initial version of the data dictionary |

## Contact

For questions or issues with the data, please open an issue in the project repository.
