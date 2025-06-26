# Judicial Vacancies Data Dictionary

This document describes the structure and content of the judicial vacancies dataset.

## Overview

The dataset contains information about current and historical judicial vacancies in the United States federal court system. The data is collected from the [US Courts website](https://www.uscourts.gov/judges-judgeships/judicial-vacancies).

## Data Fields

### Core Fields

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|----------|
| `seat` | Integer | Unique identifier for the judicial seat | 1 |
| `court` | String | Name of the court where the vacancy exists | "9th Circuit" |
| `vacancy_date` | Date | Date when the vacancy occurred | 2025-01-15 |
| `nominee` | String | Name of the nominee (if any) | "John Smith" |
| `nomination_date` | Date | Date when the nomination was made | 2025-03-10 |
| `confirmation_date` | Date | Date when the nominee was confirmed (if applicable) | 2025-05-20 |
| `vacancy_days` | Integer | Number of days the position has been vacant | 125 |
| `nomination_days` | Integer | Number of days from vacancy to nomination | 54 |
| `confirmation_days` | Integer | Number of days from nomination to confirmation | 71 |

### Court Information

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|----------|
| `court_type` | String | Type of court (e.g., Circuit, District) | "Circuit" |
| `circuit` | String | Circuit number (for Circuit Courts) | "9" |
| `state` | String | State where the court is located | "California" |
| `district` | String | District number (for District Courts) | "Northern" |

### Nomination Status

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|----------|
| `status` | String | Current status of the nomination | "Confirmed" |
| `hearing_date` | Date | Date of confirmation hearing | 2025-04-15 |
| `committee_vote_date` | Date | Date of committee vote | 2025-04-30 |
| `full_senate_vote_date` | Date | Date of full Senate vote | 2025-05-20 |
| `vote_yea` | Integer | Number of 'Yea' votes | 65 |
| `vote_nay` | Integer | Number of 'Nay' votes | 35 |

### Judge Information (for filled positions)

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

## Data Quality

- **Completeness**: Some historical records may be incomplete
- **Timeliness**: Data is updated regularly, but there may be a delay in updates
- **Accuracy**: While we strive for accuracy, users should verify critical information

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
