"""
Credit Decision Bias & Data-Drift Audit
=======================================
An AI-governance portfolio project by Femi Fayinminu.

Audits an automated credit-decision model (trained on the UCI German Credit
dataset) for two of the risks an AI governance analyst is paid to catch:

  1. FAIRNESS / BIAS  - does the model approve some demographic groups far less
     often than others?  (disparate impact / four-fifths rule, demographic
     parity, equal opportunity)
  2. DATA DRIFT       - have the model's inputs shifted away from what it was
     built on?  (Population Stability Index), plus a simulated "next quarter"
     batch so you can see the monitor fire.

Everything is computed from first principles (no black-box fairness library) so
the logic is transparent and defensible in an interview.

Governance framing (NIST AI RMF): this script operationalises MEASURE (quantify
fairness + drift) and feeds MANAGE (thresholds that trigger review).

Outputs (feed the Power BI "Model Governance & Drift Dashboard"):
  outputs/data_quality_report.csv
  outputs/fairness_metrics.csv
  outputs/drift_metrics.csv
and a plain-language summary to the console.

Usage:   python credit_bias_drift_audit.py
Requires: pandas, numpy, scikit-learn   (see requirements.txt)
Data:    UCI Statlog (German Credit Data), auto-downloaded; offline fallback noted.
"""

from __future__ import annotations
import os
import sys
import io
import urllib.request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RNG = 42
OUTDIR = "outputs"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age_years", "other_installment_plans", "housing", "existing_credits",
    "job", "num_liable", "telephone", "foreign_worker", "credit_label",
]

# Protected attributes to audit (privileged vs unprivileged group).
# We use the two well-studied, robustly-sized attributes for this dataset.
PROTECTED = {
    "sex":       {"privileged": "male",        "unprivileged": "female"},
    "age_group": {"privileged": "25_and_over", "unprivileged": "under_25"},
}


# --- 1. Load -----------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the space-delimited German Credit data (coded categoricals)."""
    try:
        with urllib.request.urlopen(DATA_URL, timeout=30) as r:
            raw = r.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None, names=COLUMNS)
    except Exception as e:  # offline / URL moved
        if os.path.exists("german.data"):
            df = pd.read_csv("german.data", sep=r"\s+", header=None, names=COLUMNS)
        else:
            sys.exit(
                f"Could not download the dataset ({e}).\n"
                "Download 'german.data' from "
                "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data "
                "and place it beside this script, then re-run."
            )
    # Target: 1 = good credit (approve), 2 = bad credit (decline) -> 1/0
    df["approved_truth"] = (df["credit_label"] == 1).astype(int)
    return df


# --- 2. Protected attributes -------------------------------------------------
def add_protected_attributes(df: pd.DataFrame) -> pd.DataFrame:
    sex_map = {"A91": "male", "A92": "female", "A93": "male",
               "A94": "male", "A95": "female"}
    df["sex"] = df["personal_status_sex"].map(sex_map)
    df["age_group"] = np.where(df["age_years"] < 25, "under_25", "25_and_over")
    # retained for the data-quality report / drift, not used as a model feature
    df["foreign_worker_flag"] = np.where(
        df["foreign_worker"] == "A201", "foreign", "domestic")
    return df


# --- 3. Data quality ---------------------------------------------------------
def data_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{
        "column": c,
        "dtype": str(df[c].dtype),
        "missing_count": int(df[c].isna().sum()),
        "missing_pct": round(df[c].isna().mean() * 100, 2),
        "n_unique": int(df[c].nunique()),
    } for c in df.columns]
    return pd.DataFrame(rows)


# --- 4. Train the decision model --------------------------------------------
def train_model(df: pd.DataFrame):
    """Train a logistic-regression approval model.

    Note: we deliberately DROP the direct sex column from the features. A key
    governance finding is whether disparate impact appears anyway, through
    proxy variables - which is why auditing outcomes (not just inputs) matters.
    """
    drop = ["credit_label", "approved_truth", "sex", "age_group",
            "foreign_worker_flag", "personal_status_sex"]
    X = df.drop(columns=drop)
    y = df["approved_truth"]
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = X.select_dtypes(include="number").columns.tolist()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    model = Pipeline([("pre", pre),
                      ("clf", LogisticRegression(max_iter=1000))])
    idx = np.arange(len(df))
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, idx, test_size=0.30, random_state=RNG, stratify=y)
    model.fit(Xtr, ytr)
    test = df.iloc[ite].copy()
    test["prediction"] = model.predict(Xte)   # 1 = model approves
    reference = df.iloc[itr].copy()            # training distribution baseline
    return model, test, reference


# --- 5. Fairness metrics -----------------------------------------------------
def _r(x, n=3):
    return round(float(x), n) if x == x else None  # None for NaN


def fairness(test: pd.DataFrame) -> pd.DataFrame:
    out = []
    for attr, roles in PROTECTED.items():
        priv, unpriv = roles["privileged"], roles["unprivileged"]
        stat = {}
        for grp in (priv, unpriv):
            g = test[test[attr] == grp]
            pos = g[g["approved_truth"] == 1]   # truly creditworthy
            neg = g[g["approved_truth"] == 0]
            stat[grp] = {
                "n": len(g),
                "sel": g["prediction"].mean() if len(g) else np.nan,
                "tpr": pos["prediction"].mean() if len(pos) else np.nan,
                "fpr": neg["prediction"].mean() if len(neg) else np.nan,
            }
            out.append({
                "protected_attribute": attr, "group": grp,
                "role": "privileged" if grp == priv else "unprivileged",
                "n": len(g), "selection_rate": _r(stat[grp]["sel"]),
                "true_positive_rate": _r(stat[grp]["tpr"]),
                "false_positive_rate": _r(stat[grp]["fpr"]),
            })
        di = stat[unpriv]["sel"] / stat[priv]["sel"] if stat[priv]["sel"] else np.nan
        dp = stat[priv]["sel"] - stat[unpriv]["sel"]
        eo = stat[priv]["tpr"] - stat[unpriv]["tpr"]
        flag = ("FAIL (<0.80)" if di < 0.80 else
                "WATCH (0.80-0.90)" if di < 0.90 else "PASS")
        out.append({
            "protected_attribute": attr, "group": "[SUMMARY]",
            "role": f"{unpriv} vs {priv}", "n": len(test),
            "disparate_impact_ratio": _r(di), "demographic_parity_diff": _r(dp),
            "equal_opportunity_diff": _r(eo), "four_fifths_rule": flag,
        })
    return pd.DataFrame(out)


# --- 6. Data drift (PSI) -----------------------------------------------------
def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for one feature (numeric or categorical)."""
    if not pd.api.types.is_numeric_dtype(ref):
        cats = ref.dropna().unique()
        r = ref.value_counts(normalize=True).reindex(cats).fillna(0)
        c = cur.value_counts(normalize=True).reindex(cats).fillna(0)
    else:
        edges = np.unique(np.quantile(ref.dropna().to_numpy(), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:          # low-cardinality numeric -> treat as categorical
            r = ref.value_counts(normalize=True)
            c = cur.value_counts(normalize=True).reindex(r.index).fillna(0)
        else:
            edges[0], edges[-1] = -np.inf, np.inf
            r = pd.cut(ref, edges).value_counts(normalize=True).sort_index()
            c = pd.cut(cur, edges).value_counts(normalize=True).sort_index()
    r, c = r.replace(0, 1e-6), c.replace(0, 1e-6)
    return float(((c - r) * np.log(c / r)).sum())


def simulate_next_batch(test: pd.DataFrame) -> pd.DataFrame:
    """A 'next quarter' production batch with injected drift (shows the monitor firing)."""
    sim = test.copy()
    rng = np.random.default_rng(RNG)
    sim["credit_amount"] = (sim["credit_amount"] * rng.normal(1.35, 0.10, len(sim))).round()
    sim["age_years"] = (sim["age_years"] - rng.integers(3, 8, len(sim))).clip(lower=18)
    sim["duration_months"] = (sim["duration_months"] * 1.20).round()
    return sim


def _band(p: float) -> str:
    return ("stable (<0.1)" if p < 0.1 else
            "moderate (0.1-0.2)" if p < 0.2 else "SIGNIFICANT (>=0.2)")


def drift(reference: pd.DataFrame, live: pd.DataFrame, nxt: pd.DataFrame) -> pd.DataFrame:
    feats = ["duration_months", "credit_amount", "age_years",
             "installment_rate", "checking_status", "savings_status"]
    rows = []
    for f in feats:
        p_live, p_next = psi(reference[f], live[f]), psi(reference[f], nxt[f])
        rows.append({
            "feature": f,
            "psi_reference_vs_live": round(p_live, 4), "live_status": _band(p_live),
            "psi_reference_vs_next_batch": round(p_next, 4),
            "next_batch_status": _band(p_next),
        })
    return pd.DataFrame(rows)


# --- main --------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    df = add_protected_attributes(load_data())

    data_quality(df).to_csv(f"{OUTDIR}/data_quality_report.csv", index=False)
    model, test, reference = train_model(df)
    fair = fairness(test)
    fair.to_csv(f"{OUTDIR}/fairness_metrics.csv", index=False)
    nxt = simulate_next_batch(test)
    dr = drift(reference, test, nxt)
    dr.to_csv(f"{OUTDIR}/drift_metrics.csv", index=False)

    print("=" * 72)
    print("CREDIT DECISION  -  BIAS & DRIFT AUDIT")
    print("=" * 72)
    print(f"Records: {len(df)}   model: LogisticRegression   test set: {len(test)}")
    print("\nFAIRNESS  (four-fifths / 80% rule on model approvals)")
    for _, r in fair[fair["group"] == "[SUMMARY]"].iterrows():
        print(f"  - {r['protected_attribute']:<12} "
              f"DI={r['disparate_impact_ratio']}  "
              f"DemParity_diff={r['demographic_parity_diff']}  "
              f"EqOpp_diff={r['equal_opportunity_diff']}  -> {r['four_fifths_rule']}")
    print("\nDATA DRIFT  (Population Stability Index vs training reference)")
    for _, r in dr.iterrows():
        print(f"  - {r['feature']:<17} live={r['psi_reference_vs_live']:<7} "
              f"({r['live_status']})   "
              f"next_batch={r['psi_reference_vs_next_batch']:<7} ({r['next_batch_status']})")
    print(f"\nCSVs written to ./{OUTDIR}/  -  load these into the Power BI dashboard.")
    print("=" * 72)


if __name__ == "__main__":
    main()
