"""Load the raw data with the fixes that notebook 01 found and justified.

Only fixes shown necessary by a notebook output live here; each names the
notebook section that found it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get("PROMO_UPLIFT_DATA", Path(__file__).resolve().parents[1] / "data" / "raw"))
HILLSTROM = DATA / "hillstrom_2008-03-20.csv"

# The three e-mail groups, control first, in the order results are reported.
ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
PRE_TREATMENT = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
OUTCOMES = ["visit", "conversion", "spend"]


def load_hillstrom(path: Path | str = HILLSTROM) -> pd.DataFrame:
    """Read the e-mail test, one row per customer, with notebook 01's fixes.

    - `zip_code`: `Surburban` renamed to `Suburban` (01, section 3). The
      only change to any value.
    - No rows dropped: the identical rows are kept (01, section 2).
    - `segment` as an ordered categorical, control first.
    """
    df = pd.read_csv(path)
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})
    df["segment"] = pd.Categorical(df["segment"], categories=ARMS, ordered=True)
    return df
