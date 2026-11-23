# Executive Summary — Credit Model Bias & Drift Audit

**Prepared by:** Femi Fayinminu  ·  **Audience:** Risk / Compliance leadership (non-technical)
**Scope:** Fairness and data-drift review of an automated credit-approval model.

## Why this matters
An automated model decides who gets credit. If it approves some groups far less often than others,
the bank carries legal and reputational risk; if the incoming data has shifted away from what the
model was trained on, the model is making today's decisions on yesterday's assumptions. This review
measured both.

## What we found

**1. Fairness — one green, one to watch.**
- **Sex:** approvals are effectively at parity (disparate-impact ratio 0.93, inside the 80% rule).
  Notably, the model was never given sex as an input, and it did **not** back-door the bias through
  proxies — a good result.
- **Age:** applicants **under 25 are approved at ~86% the rate of older applicants** (disparate-impact
  ratio 0.86). The gap holds **even among genuinely creditworthy young applicants**. This is above the
  0.80 legal-red-flag line but inside the zone that requires a documented justification and ongoing monitoring.

**2. Drift monitoring — working as designed.**
- On current data, the monitor is quiet (no meaningful drift).
- On a simulated next-quarter batch, it **correctly flagged** significant shifts in loan duration, age,
  and credit amount — the signal a team needs to pause and re-check the model before it keeps deciding.

## Risk if ignored
- **Age gap unexamined** → potential discrimination exposure and no audit trail of a justification.
- **No drift gate** → silent model decay; approvals drift off-target without anyone noticing until losses appear.

## Recommended actions
1. Require a documented business justification for the age gap; track disparate impact as a monitored KPI.
2. Run the drift check on every new batch; a PSI of 0.2+ on any decision feature triggers human review
   and a retraining decision.
3. Audit **outcomes**, not just inputs — bias can appear without the protected attribute being a feature.

## Framework alignment
Maps to the **NIST AI RMF** Measure function (quantifying fairness and drift) and Manage function
(thresholds that trigger review), and to the fairness/transparency expectations in **ISO/IEC 42001**
and the **EU AI Act** high-risk requirements for data governance and accuracy monitoring.
