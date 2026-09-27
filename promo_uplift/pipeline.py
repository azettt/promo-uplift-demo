"""Rebuild the tables the demo reads, from the raw file, without a notebook.

    python -m promo_uplift.pipeline                 # writes to data/
    python -m promo_uplift.pipeline --out some/dir  # writes elsewhere

(run from the repository root; needs scikit-learn and the raw CSV, see the README)

Notebooks 04 and 05 are the record of how the buyer-type rule and the
allocation were chosen. This script is the part that runs again when the
data changes, in the order of the course pipeline it follows:

    1. ingest    load the raw file with notebook 01's one fix
    2. prepare   add the buyer type, the only attribute that changed the
                 effect (notebook 04, section 3)
    3. split     the same 50/50 split as notebook 04, stratified by group
    4. train     the rule's "model": the visit effect of each e-mail within
                 each buyer type, measured on the training half
    5. evaluate  the same effects on the held-out half
    6. save      the three tables the demo and notebook 05 read

There is no pickle: the rule is six numbers (three buyer types, two
e-mails), saved as a table, and the demo solves the allocation itself.
`tests/test_pipeline.py` checks that the output equals the notebooks'.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import load_hillstrom

OUTPUTS = Path(__file__).resolve().parents[1] / "data"

OUTCOME = "visit"      # visits drive the targeting (notebook 01, section 6)
SPLIT_SEED = 2026      # notebook 04, section 5
TYPES = ["both", "men's only", "women's only"]
ARMS = {1: "Mens E-Mail", 2: "Womens E-Mail"}  # group code -> name; 0 is no e-mail


def add_buyer_type(df: pd.DataFrame) -> pd.DataFrame:
    """Which product line a customer bought from in the past year."""
    df = df.copy()
    men_only = df["mens"].eq(1) & df["womens"].eq(0)
    women_only = df["mens"].eq(0) & df["womens"].eq(1)
    df["buyer_type"] = np.select([men_only, women_only], ["men's only", "women's only"], "both")
    df["arm"] = df["segment"].cat.codes  # 0 no e-mail, 1 men's, 2 women's
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Half to learn the effects on, half to value the plans on."""
    return train_test_split(df, test_size=0.5, random_state=SPLIT_SEED, stratify=df["arm"])


def group_effects(half: pd.DataFrame, label: str) -> pd.DataFrame:
    """Effect of each e-mail on the visit rate within each buyer type,
    with its standard error (difference of two means)."""
    rows = []
    for buyer_type, g in half.groupby("buyer_type"):
        control = g.loc[g["arm"] == 0, OUTCOME]
        for code, name in ARMS.items():
            sent = g.loc[g["arm"] == code, OUTCOME]
            rows.append({
                "half": label, "buyer_type": buyer_type, "arm": name, "customers": len(g),
                "effect": sent.mean() - control.mean(),
                "se": np.sqrt(sent.var() / len(sent) + control.var() / len(control)),
                "control_rate": control.mean(),
            })
    return pd.DataFrame(rows)


def demo_table(effects: pd.DataFrame, held_out: pd.DataFrame) -> pd.DataFrame:
    """One row per buyer type and e-mail: what the demo plans with (training
    effect) and what it reports (held-out visit and purchase effects)."""
    train = effects[effects["half"] == "train"].set_index(["buyer_type", "arm"])
    visit = held_out.groupby(["buyer_type", "arm"])["visit"].mean().unstack()
    purchase = held_out.groupby(["buyer_type", "arm"])["conversion"].mean().unstack()
    sizes = held_out["buyer_type"].value_counts()
    rows = []
    for t in TYPES:
        for code, name in ARMS.items():
            rows.append({
                "buyer_type": t, "arm": name, "customers_held_out": int(sizes[t]),
                "uplift_visit_train": train.loc[(t, name), "effect"],
                "se_train": train.loc[(t, name), "se"],
                "uplift_visit_held_out": visit.loc[t, code] - visit.loc[t, 0],
                "uplift_purchase_held_out": purchase.loc[t, code] - purchase.loc[t, 0],
            })
    return pd.DataFrame(rows)


def run(out: Path = OUTPUTS) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)

    df = add_buyer_type(load_hillstrom())                          # 1, 2
    train, test = split(df)                                        # 3
    effects = pd.concat([group_effects(train, "train"),            # 4
                         group_effects(test, "test")], ignore_index=True)  # 5
    held_out = test[["arm", "buyer_type", OUTCOME, "conversion", "spend", "zip_code", "channel"]]
    held_out = held_out.assign(row=test.index)

    paths = {                                                      # 6
        "effects": out / "uplift_by_buyer_type.csv",
        "held_out": out / "test_half.csv",
        "demo": out / "demo_group_effects.csv",
    }
    effects.to_csv(paths["effects"], index=False)
    held_out.to_csv(paths["held_out"], index=False)
    # the demo table is built from the saved files, as notebook 05 builds it,
    # so the numbers agree to the last digit whichever of the two wrote them
    demo = demo_table(pd.read_csv(paths["effects"]), pd.read_csv(paths["held_out"]))
    demo.to_csv(paths["demo"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    for name, path in run(args.out).items():
        print(f"{name:9s} {path}")


if __name__ == "__main__":
    main()
