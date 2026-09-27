# Which e-mail should this customer get?

A small demo of the allocation in **Promotion Uplift and Bandits**. Set an
e-mail budget, how many men's e-mails are available and what one costs,
describe a customer, and the app re-solves the plan: which e-mail that
customer's group gets, why, and how the plan compares with ranking
customers by response.

Project write-up: https://azaryamarcello.vercel.app/projects/promo-uplift

![The demo](demo.webp)

## What it shows

On a randomized e-mail test of 64,000 customers, the effect of each e-mail
differed by one thing only: what a customer had bought before (men's
only, women's only, or both). So targeting is a three-group rule, and
the allocation is an integer program over those groups: each customer
gets at most one e-mail, within the budget and a limit per e-mail.

With a budget alone, ranking by response and cutting at the budget gives
the program's plan exactly. When the men's e-mail is scarce, ranking
gives it to whoever responds most, when it should go to whoever loses
most without it. In the default scenario (32,000 e-mails, men's capped
at 16,000) the program predicts 2,515 extra visits against ranking's
1,793. Budgets, limits and costs are scenarios, not real figures.

## No pickle, on purpose

The "model" is six numbers: the effect of each of two e-mails in each of
three buyer types, measured on the training half of the test. They sit
in `data/demo_group_effects.csv`, readable and diffable, beside the
effects measured on the held-out half. The app reads that table and runs
the solver (PuLP with its bundled CBC) on every change. A pickle earns
its place when a model has many fitted parameters, such as a gradient
boosted ensemble, and then with the library versions pinned.

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

Tests (the default scenario must reproduce the project's figures):

```
pip install pytest
python -m pytest
```

## Rebuild the table

`promo_uplift/pipeline.py` rebuilds `data/demo_group_effects.csv` from the
raw test: load, add the buyer type, split 50/50 with a fixed seed,
measure each e-mail's effect per buyer type on each half. It needs
scikit-learn and the raw CSV, which this repository does not include:

```
pip install scikit-learn
mkdir -p data/raw
curl -o data/raw/hillstrom_2008-03-20.csv "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
python -m promo_uplift.pipeline
```

## Where things are

| Path | What |
|---|---|
| `app.py` | The Streamlit app |
| `promo_uplift/allocate.py` | The integer program and the greedy rules |
| `promo_uplift/pipeline.py`, `data.py` | Raw file to the demo's table |
| `data/demo_group_effects.csv` | The six effects the app plans with |
| `tests/` | The allocation against known answers, and the app's default scenario |

## Data

The MineThatData e-mail test, released by Kevin Hillstrom in 2008. Only
the aggregated effects table is included here.

Code under the MIT License.
