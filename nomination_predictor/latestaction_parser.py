import re
from typing import Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# 1.  Pre-compiled regexes
# ---------------------------------------------------------------------------
RE_YEA_NAY   = re.compile(r"yea[-\s]*nay vote\.?\s*(\d+)\s*-\s*(\d+)", re.I)
RE_YEA_ONLY  = re.compile(r"yea[-\s]*nay vote\.?\s*(\d+)(?!\s*-\s*\d+)", re.I)
RE_BULLET    = re.compile(r"^\s*-\s*(\d+)\.", re.I)                      # "- 45. Record…"
RE_RECORD_NO = re.compile(r"record vote number:\s*(\d+)", re.I)

# ---------------------------------------------------------------------------
def _parse_vote(text: str) -> Tuple[Optional[int], Optional[int], Optional[int],
                                    bool, bool]:
    """
    Extract yea, nay, record#, was_voice, was_unanimous from one action string.
    Returns tuple (yea, nay, record_num, voice_vote?, unanimous?).
    """
    if not isinstance(text, str):
        return None, None, None, False, False

    t = text.lower()

    # voice vote?
    voice = "voice vote" in t

    # yea / nay
    yea = nay = None
    m = RE_YEA_NAY.search(t)
    if m:
        yea, nay = int(m.group(1)), int(m.group(2))
    else:
        m = RE_YEA_ONLY.search(t)
        if m:
            yea = int(m.group(1))
        else:
            m = RE_BULLET.match(t)
            if m:
                yea = int(m.group(1))

    # record number
    m = RE_RECORD_NO.search(t)
    rec_num = int(m.group(1)) if m else None

    # unanimous?
    unanimous = ("unanimous" in t) or (yea and yea >= 100) or (yea and nay == 0)

    return yea, nay, rec_num, voice, unanimous


def _classify_action(text: str) -> str:
    """Return categorical label of latest action."""
    if not isinstance(text, str):
        return "unknown"

    t = text.lower()
    if "confirmed" in t:
        return "confirmed"
    if "withdraw" in t:
        return "withdrawn"
    if "returned" in t:
        return "returned"
    return "other"


# ---------------------------------------------------------------------------
def enrich_latest_action(df: pd.DataFrame,
                         col: str = "latestaction_text") -> pd.DataFrame:
    """
    Add columns:
      • latest_action_taken   (confirmed / returned / withdrawn / other)
      • yea_votes             (int or NaN)
      • nay_votes             (int or NaN)
      • record_vote_number    (int or NaN)
      • was_voice_vote        (bool)
      • was_unanimous_decision(bool)

    The function operates on a *copy* and returns it.
    """
    yea, nay, rno, voice, unanimity = [], [], [], [], []
    action_labels = []

    for txt in df[col]:
        action_labels.append(_classify_action(txt))
        y, n, rv, vv, un = _parse_vote(txt)
        yea.append(y)
        nay.append(n)
        rno.append(rv)
        voice.append(vv)
        unanimity.append(un)

    new_df = df.copy()
    new_df["latest_action_taken"]    = action_labels
    new_df["yea_votes"]              = pd.Series(yea, dtype="Int64")
    new_df["nay_votes"]              = pd.Series(nay, dtype="Int64")
    new_df["record_vote_number"]     = pd.Series(rno, dtype="Int64")
    new_df["was_voice_vote"]         = voice
    new_df["was_unanimous_decision"] = unanimity

    return new_df
