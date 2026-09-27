"""Which e-mail for this customer? A small demo of project C's allocation.

Loads the saved group effects (data/demo_group_effects.csv), re-solves
the grouped integer program for the budget and limits chosen in the
sidebar, and says in plain words what the described customer gets and
why. No model is refitted: the buyer-type rule is six numbers in a table.

Run from the repository root:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from promo_uplift.allocate import greedy, solve_groups  # noqa: E402

ARMS = ["Mens E-Mail", "Womens E-Mail"]
LABEL = {"Mens E-Mail": "men's e-mail", "Womens E-Mail": "women's e-mail"}
TYPES = ["both", "men's only", "women's only"]


@st.cache_data
def load():
    return pd.read_csv(ROOT / "data" / "demo_group_effects.csv")


def buyer_type(mens: int, womens: int) -> str:
    if mens and womens:
        return "both"
    return "men's only" if mens else "women's only"


def plan(eff, budget, men_cap, men_cost):
    sizes = eff.drop_duplicates("buyer_type").set_index("buyer_type")["customers_held_out"].loc[TYPES]
    upl = {(r.buyer_type, r.arm): r.uplift_visit_train for r in eff.itertuples()}
    cost = {"Mens E-Mail": men_cost, "Womens E-Mail": 1.0}
    cap = {"Mens E-Mail": men_cap} if men_cap < sizes.sum() else None
    counts, obj, status = solve_groups(sizes.to_dict(), upl, budget, cost=cost, capacity=cap)
    table = pd.Series(counts).unstack().loc[TYPES, ARMS].round()
    # greedy on one row per customer, for comparison
    u = [[upl[t, a] for a in ARMS] for t in TYPES for _ in range(int(sizes[t]))]
    types = [t for t in TYPES for _ in range(int(sizes[t]))]
    g = greedy(u, budget, cost=[men_cost, 1.0], capacity={0: men_cap} if cap else None)
    # count in plain Python; 32,000 single-cell .loc updates took seconds
    counted = Counter((t, ARMS[a]) for t, a in zip(types, g) if a >= 0)
    gtab = pd.DataFrame([[counted[t, a] for a in ARMS] for t in TYPES], index=TYPES, columns=ARMS)
    g_value = sum(gtab.loc[t, a] * upl[t, a] for t in TYPES for a in ARMS)
    return table, obj, status, gtab, g_value, sizes


def main():
    st.set_page_config(page_title="Which e-mail?", layout="wide")
    eff = load()
    n = int(eff.drop_duplicates("buyer_type")["customers_held_out"].sum())

    st.title("Which e-mail should this customer get?")
    st.caption("MineThatData e-mail test (Hillstrom, 2008), 32,000 held-out customers. "
               "Effects are on website visits within two weeks. Budgets, limits and costs are scenarios.")

    with st.sidebar:
        st.header("Scenario")
        budget = st.slider("E-mail budget (units)", 0, 2 * n, n, step=1000)
        men_cap = st.slider("Men's e-mails available", 0, n, n // 2, step=1000)
        men_cost = st.select_slider("Cost of a men's e-mail (women's = 1)", [1.0, 1.5, 2.0], value=1.0)
        st.header("Customer")
        mens = st.checkbox("Bought men's merchandise last year", True)
        womens = st.checkbox("Bought women's merchandise last year", False)
        if not (mens or womens):
            st.warning("Every customer in this data bought one or the other; pick at least one.")
            st.stop()
        bt = buyer_type(int(mens), int(womens))

    table, obj, status, gtab, g_value, sizes = plan(eff, budget, men_cap, men_cost)
    mine = eff[eff["buyer_type"] == bt].set_index("arm")

    left, right = st.columns(2)
    with left:
        st.subheader(f"Buyer type: {bt}")
        st.write("Predicted extra chance of a visit, from the training half "
                 "(90% interval), and what the held-out half measured:")
        show = pd.DataFrame({
            "predicted": mine["uplift_visit_train"].map("{:+.1%}".format),
            "90% interval": [f"{lo:+.1%} to {hi:+.1%}" for lo, hi in
                             zip(mine["uplift_visit_train"] - 1.645 * mine["se_train"],
                                 mine["uplift_visit_train"] + 1.645 * mine["se_train"])],
            "held out": mine["uplift_visit_held_out"].map("{:+.1%}".format),
        }).rename(index=LABEL)
        st.table(show)
    with right:
        st.subheader("What the plan gives this group")
        if status != "Optimal":
            st.error(f"No plan: the solver reports {status}.")
            st.stop()
        got = table.loc[bt]
        size = int(sizes[bt])
        parts = [f"{int(got[a]):,} get the {LABEL[a]}" for a in ARMS if got[a] > 0]
        none = size - int(got.sum())
        if none:
            parts.append(f"{none:,} get no e-mail")
        st.write(f"Of the {size:,} customers of this type: " + "; ".join(parts) + ".")
        best = mine["uplift_visit_train"].idxmax()
        other = [a for a in ARMS if a != best][0]
        loss = mine["uplift_visit_train"].max() - mine["uplift_visit_train"].min()
        if got.sum() == 0:
            why = "The budget is spent on customers whose predicted gain is larger."
        elif got[other] == 0:
            why = f"This type responds most to the {LABEL[best]}, and the plan could afford it."
        elif got[best] > 0:
            why = (f"This type responds most to the {LABEL[best]}, but there is not enough of it: "
                   f"{int(got[best]):,} get it and {int(got[other]):,} get the {LABEL[other]}, "
                   f"which still adds {mine.loc[other, 'uplift_visit_train']:.1%}.")
        else:
            why = (f"This type responds a little more to the {LABEL[best]}, but would lose only "
                   f"{loss:.1%} with the {LABEL[other]}. The {LABEL[best]} is scarce or costs more, "
                   "so it goes to customers who would lose more without it.")
        st.info(why)
        st.caption("Customers of one type are identical to the model, so the plan says how many of "
                   "them get each e-mail, not which ones.")

    st.subheader("The whole plan")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Integer program**")
        st.dataframe(table.rename(columns=LABEL).astype(int))
        st.metric("Predicted extra visits", f"{obj:,.0f}")
    with c2:
        st.write("**Greedy (rank and cut)**")
        st.dataframe(gtab.rename(columns=LABEL))
        st.metric("Predicted extra visits", f"{g_value:,.0f}", delta=f"{g_value - obj:,.0f} against the program")
    st.caption("Rank and cut gives a scarce e-mail to whoever responds most. A greedy that asks instead "
               "who loses most without it matched the integer program in every scenario tested.")


if __name__ == "__main__":
    main()
