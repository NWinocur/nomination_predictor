import re
from typing import List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
def _extract_service_indices(cols: List[str]) -> List[str]:
    """
    Return the distinct numeric suffixes '1', '2', … that occur in
    confirmation_date_X columns (robust to any total count).
    """
    pat = re.compile(r"confirmation_date_(\d+)")
    return sorted({m.group(1) for c in cols if (m := pat.match(c))})


def compress_federal_service_features(
    df: pd.DataFrame,
    received_col: str = "receiveddate",
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Adds two columns to *df*:

        • days_federal_service_experience  – total confirmed-to-termination
                                              days across all prior federal
                                              judgeships for that nominee
        • prior_fed_roles_count            – how many prior federal roles

    Logic per row
    -------------
    - For every suffix N where **confirmation_date_N** exists:
        start = confirmation_date_N
        end   = termination_date_N  (if NaT, fall back to row[received_col])
        if start is valid → count role and add (end-start).days if end ≥ start
    - If no valid start dates → count = <NA>, days = <NA>

    Parameters
    ----------
    received_col  : column used as fallback 'end' if termination missing
    drop_original : if True, delete the wide service_*_N columns afterwards

    Returns
    -------
    df  (the same DataFrame object, modified in-place for convenience)
    """
    # --- find all service indices dynamically ------------------------------
    indices = _extract_service_indices(df.columns)
    if not indices:      # nothing to do
        df["days_federal_service_experience"] = pd.NA
        df["prior_fed_roles_count"] = pd.NA
        return df

    # --- compute per-row ----------------------------------------------------
    def _row_agg(row) -> Tuple[int, int]:
        total_days = 0
        cnt        = 0
        for n in indices:
            start = pd.to_datetime(row.get(f"confirmation_date_{n}"), errors="coerce")
            if pd.isna(start):
                continue
            end = pd.to_datetime(row.get(f"termination_date_{n}"), errors="coerce")
            if pd.isna(end):
                end = pd.to_datetime(row.get(received_col), errors="coerce")
            if pd.isna(end) or end < start:
                continue
            total_days += (end - start).days
            cnt += 1
        return total_days if cnt else pd.NA, cnt if cnt else pd.NA

    out = df.apply(lambda r: _row_agg(r), axis=1, result_type="expand")
    out.columns = ["days_federal_service_experience", "prior_fed_roles_count"]
    df[["days_federal_service_experience", "prior_fed_roles_count"]] = out

    # --- optionally drop the wide service columns --------------------------
    if drop_original:
        patt = re.compile(
            r"(court_name|court_type|appointment_title|appointing_president|"
            r"party_of_appointing_president|nomination_date|confirmation_date|"
            r"commission_date|termination_date)_\d+"
        )
        df.drop(columns=[c for c in df.columns if patt.match(c)], inplace=True)

    return df
