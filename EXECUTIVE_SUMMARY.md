# Executive Summary: Credit Model Bias and Drift Audit

**By:** Femi Fayinminu  ·  **For:** risk and compliance leaders (non-technical)
**Scope:** a fairness and data-drift review of an automated credit-approval model.

## Why it matters
A model decides who gets credit. If it approves some groups much less often than others, the bank carries legal and reputation risk. If the incoming data has moved away from what the model was trained on, the model is deciding today on old assumptions. This review checked both.

## What we found

**1. Fairness: one is fine, one to watch.**
- **Sex:** approvals are close to even (ratio 0.93, inside the 80 percent rule). The model was never given sex, and it did not pick up the bias through other columns, which is a good sign.
- **Age:** people under 25 are approved at about 86 percent of the rate of older people (ratio 0.86). The gap holds even among young people who are actually creditworthy. It is above the 0.80 legal line but in the range that needs a written reason and ongoing checks.

**2. Drift monitor: working.**
- On current data, the monitor is quiet (no real drift).
- On a made-up next-quarter batch, it flagged big moves in loan length, age and credit amount, which is the sign a team needs to pause and recheck the model.

## The risk if we ignore it
- Leaving the age gap unchecked: possible discrimination risk, with no record of a reason.
- No drift gate: the model quietly goes off target and no one notices until losses show up.

## What to do
1. Ask for a clear business reason for the age gap, and track disparate impact as a number you watch.
2. Run the drift check on every new batch. A PSI of 0.2 or more on any decision column sends it to a person and a retrain decision.
3. Check the outcomes, not just the inputs. Bias can show up without the protected trait being an input.

## How it lines up with the rules
It matches the NIST AI RMF Measure part (put numbers on fairness and drift) and Manage part (thresholds that send a case for review), and the fairness and transparency parts of ISO/IEC 42001 and the EU AI Act high-risk rules for data handling and accuracy checks.
