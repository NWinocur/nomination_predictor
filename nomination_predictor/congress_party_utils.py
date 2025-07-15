"""
congress_party_utils.py
---------------------------------
• party_at_date(d)            -> ("D"|"R", "D"|"R")  # (house_party, senate_party)
• add_alignment_flags(df)     -> df with 8 Bool columns
"""

from datetime import date
import logging
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  HOUSE & SENATE PARTY CONTROL — 95th Congress (1977‑) forward
#     tuple = (start_date, end_date, house_party, senate_party)
# ---------------------------------------------------------------------------
CONTROL: List[Tuple[date, date, str, str]] = [
    # 95th–96th  (Carter)
    (date(1977,1,3),  date(1981,1,3),  "D", "D"),
    # 97th–99th  (Reagan, split Senate R / House D)
    (date(1981,1,3),  date(1987,1,3),  "D", "R"),
    # 100th–102nd (1987‑1993) unified D
    (date(1987,1,3),  date(1995,1,3),  "D", "D"),
    # 104th–106th (1995‑2001) unified R
    (date(1995,1,3),  date(2001,1,3),  "R", "R"),
    # 107th Congress is tricky – control flips on 2001‑06‑06
    (date(2001,1,3),  date(2001,6,6),  "R", "R"),   # GOP 50‑50 with VP tie
    (date(2001,6,6),  date(2003,1,3),  "R", "D"),   # Jeffords switch
    # 108th–109th unified R (2003‑2007)
    (date(2003,1,3),  date(2007,1,3),  "R", "R"),
    # 110th–111th unified D (2007‑2011)
    (date(2007,1,3),  date(2011,1,3),  "D", "D"),
    # 112th–113th split: R House, D Senate (2011‑2015)
    (date(2011,1,3),  date(2015,1,3),  "R", "D"),
    # 114th–115th unified R (2015‑2019)
    (date(2015,1,3),  date(2019,1,3),  "R", "R"),
    # 116th split: D House, R Senate (2019‑2021)
    (date(2019,1,3),  date(2021,1,20), "D", "R"),
    # 117th unified D (tie + VP) with D House (2021‑01‑20 → 2023‑01‑03)
    (date(2021,1,20), date(2023,1,3),  "D", "D"),
    # 118th split: R House, D Senate (2023‑2025)
    (date(2023,1,3),  date(2025,1,3),  "R", "D"),
    # 119th split: R House, R Senate (2025‑2027)
    (date(2025,1,3),  date(2027,1,3),  "R", "R"),
]

# sort newest‑first for fast lookup
CONTROL.sort(key=lambda t: t[0], reverse=True)

# ---------------------------------------------------------------------------
def _lookup_parties(d: date) -> Optional[Tuple[str, str]]:
    """Return (house_party, senate_party) or None if date < 1977‑01‑03."""
    for start, end, hp, sp in CONTROL:
        if start <= d < end:
            return hp, sp
    return None


def party_at_date(dt) -> Tuple[Optional[str], Optional[str]]:
    """
    Vector‑safe wrapper: accepts pandas Timestamp / datetime / date.
    Returns tuple (house_party, senate_party) or (None, None) if out‑of‑range.
    """
    if pd.isna(dt):
        return (None, None)
    if isinstance(dt, pd.Timestamp):
        dt = dt.date()
    if not isinstance(dt, date):
        return (None, None)
    out = _lookup_parties(dt)
    if out is None:
        logger.error("Date %s not yet supported by stored legislature data", dt)
        return (None, None)
    return out


# ---------------------------------------------------------------------------
def _alignment_flags(pres_party: str,
                     house_party: Optional[str],
                     senate_party: Optional[str]) -> Tuple[Optional[bool], ...]:
    """
    Produce the four Booleans (unified, div_opp_house, div_opp_senate, fully_div)
    with <NA> when house/senate info missing.
    """
    if house_party is None or senate_party is None:
        return (pd.NA, pd.NA, pd.NA, pd.NA)
    unified          = house_party == senate_party == pres_party
    div_opp_house    = (senate_party == pres_party) and (house_party != pres_party)
    div_opp_senate   = (house_party == pres_party) and (senate_party != pres_party)
    fully_div        = (house_party != pres_party)  and (senate_party != pres_party)
    return unified, div_opp_house, div_opp_senate, fully_div


def add_alignment_flags(
    df: pd.DataFrame,
    pres_party_col: str = "pres_party",
    nom_date_col: str = "receiveddate",
    act_date_col: str = "latestaction_actiondate",
) -> pd.DataFrame:
    """
    Adds eight Boolean (nullable) columns to *df*:

        • nom_is_unified
        • nom_is_div_opp_house
        • nom_is_div_opp_senate
        • nom_is_fully_div
        • latestaction_is_unified
        • latestaction_is_div_opp_house
        • latestaction_is_div_opp_senate
        • latestaction_is_fully_div
    """
    # ensure pandas datetime
    df[nom_date_col] = pd.to_datetime(df[nom_date_col], errors="coerce")
    df[act_date_col] = pd.to_datetime(df[act_date_col], errors="coerce")

    # Vectorised evaluation
    nom_hp, nom_sp = zip(*df[nom_date_col].apply(party_at_date))
    act_hp, act_sp = zip(*df[act_date_col].apply(party_at_date))

    df["_nom_house"], df["_nom_sen"] = nom_hp, nom_sp
    df["_act_house"], df["_act_sen"] = act_hp, act_sp

    # Apply alignment logic row‑wise (fast on pandas)
    nom_flags = df.apply(
        lambda r: _alignment_flags(r[pres_party_col], r["_nom_house"], r["_nom_sen"]),
        axis=1, result_type="expand"
    )
    act_flags = df.apply(
        lambda r: _alignment_flags(r[pres_party_col], r["_act_house"], r["_act_sen"]),
        axis=1, result_type="expand"
    )

    nom_flags.columns = [
        "nom_is_unified", "nom_is_div_opp_house",
        "nom_is_div_opp_senate", "nom_is_fully_div"
    ]
    act_flags.columns = [
        "latestaction_is_unified", "latestaction_is_div_opp_house",
        "latestaction_is_div_opp_senate", "latestaction_is_fully_div"
    ]

    # Attach & drop temp cols
    df = pd.concat([df, nom_flags, act_flags], axis=1)
    df.drop(columns=["_nom_house", "_nom_sen", "_act_house", "_act_sen"], inplace=True)

    # cast to pandas BooleanDtype for proper <NA>
    bool_cols = [
        c for c in df.columns
        if c.startswith("nom_is_") or c.startswith("latestaction_is_")
    ]
    df[bool_cols] = df[bool_cols].astype("boolean")
    return df
