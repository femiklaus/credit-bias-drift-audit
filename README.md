# Credit Decision: Bias and Data-Drift Audit

A portfolio project. It audits an automated credit-decision model for two things a governance or risk analyst gets asked to check: is it fair, and has the data moved. I wrote the fairness and drift checks myself instead of using a library, so the method is easy to follow.

**By:** Femi Fayinminu, AI Governance and Security
**Write-up (Notion):** _[add link]_  ·  **Data:** UCI Statlog (German Credit Data)

## The question

A bank uses a model to approve or decline credit. Two things a review has to answer:

1. **Is it fair?** Does the model approve some groups much less often than others, even though it was never given the protected trait (sex) as an input?
2. **Is it still valid?** Has the incoming data moved away from what the model was trained on, so that yesterday's model is deciding on a different population today?

## What it does

- Trains a logistic-regression approval model. I leave sex out of the inputs on purpose, so I can test whether unfairness still shows up through other columns.
- **Fairness check** on the model's approvals across groups:
  - **Disparate impact (the 80 percent rule):** the ratio of approval rates, unprivileged over privileged.
  - **Demographic parity gap:** the difference in approval rates.
  - **Equal opportunity gap:** the difference in approval rates among people who are actually creditworthy.
- **Drift check** using the Population Stability Index (PSI) for each feature, with a made-up "next quarter" batch so you can see the monitor react.

### How it maps to the NIST AI RMF
It covers the Measure part (put a number on fairness and drift) and feeds the Manage part (set thresholds that send a case to a person): disparate impact below 0.80 fails, 0.80 to 0.90 is a watch, and a PSI of 0.2 or more is real drift.

## Run it

```bash
pip install -r requirements.txt
python credit_bias_drift_audit.py
python make_charts.py
```

The data downloads on its own. If you are offline, put `german.data` next to the script. Results go to `outputs/` and feed the charts and the Power BI dashboard.

| File | What is in it |
|---|---|
| `outputs/data_quality_report.csv` | type, missing values and number of unique values per column |
| `outputs/fairness_metrics.csv` | rates per group, plus the disparate impact and parity summary |
| `outputs/drift_metrics.csv` | PSI per feature, live batch vs the made-up next batch |

## Findings (same every run, fixed seed)

**Fairness**

| Group | Disparate impact | Parity gap | Equal opportunity gap | Result |
|---|---|---|---|---|
| Sex (female vs male) | **0.93** | 0.05 | 0.03 | Passes the 80 percent rule |
| Age (under 25 vs 25 and over) | **0.86** | 0.11 | 0.13 | **Watch** |

Even though sex is never an input, the result on sex is close to fair, and the model did not pick up the bias through other columns. Age is different: people under 25 are approved at about 86 percent of the rate of older people, and the gap holds even among young people who are actually creditworthy. It sits above the 0.80 legal line but in the range that needs a closer look and a written business reason.

**Drift.** PSI stays low on the live data (no drift, as expected). The monitor reacts on the made-up next-quarter batch: loan duration (0.64), age (0.38) and credit amount (0.21) all cross the 0.2 line, while the columns I did not change stay quiet. So it is quiet when it should be and loud when it matters.

## Dashboard

Run `python make_charts.py` to build the charts from the CSVs (saved to `charts/`):

![Fairness, disparate impact](charts/fairness_disparate_impact.png)

![Data drift, PSI](charts/drift_psi.png)

## What to do about it (the governance output)

- **Age:** ask for a clear business reason for the gap, and track disparate impact as a number you watch.
- **Drift:** run the PSI check on every new batch. A PSI of 0.2 or more on any decision column sends the batch to a person and a retrain decision before the model keeps deciding.
- **Proxies:** unfairness can show up without the protected trait being an input, so check the outcomes, not just the inputs.

## Honest limits

- The label is past credit outcomes, which may carry their own bias. A real audit would also look at where the label came from.
- The next-quarter drift is made up to show the monitor working. In real life it runs on real batches.
- I left `foreign_worker` out of the fairness view because the home group is too small (about 3.7 percent) for a steady rate. A real audit would mark it as too small to judge and keep watching.

## What this shows
Python, scikit-learn, pandas and numpy, fairness checks, PSI drift detection, data-quality checks, NIST AI RMF framing, and turning the result into controls a non-technical manager can act on.
