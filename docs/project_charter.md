\# Project Charter: Employee Attrition Analysis \& Prediction



\## Status

Analytical prototype. Not a production HR system.



\## Business problem

An HR team wants to understand which employee attributes are associated with

attrition, and whether a model can flag employees at higher risk of leaving so

HR can start supportive retention conversations earlier.



\## Cost of errors

\- False negative: a leaver is missed; lost chance to intervene, plus

&#x20; recruitment and ramp-up cost.

\- False positive: a stayer is flagged; wasted conversation and risk of unfair

&#x20; labelling.



Model selection will prioritise attrition-class recall and precision-recall

behaviour over accuracy.



\## Scope

In: EDA, leakage-safe pipeline, DummyClassifier / Logistic Regression /

Random Forest / Gradient Boosting comparison, untouched test evaluation,

interpretation, README. Optional: Streamlit demo.



Out: deployment, causal claims, real HR decision-making.



\## Success criteria

1\. Dataset shape, class counts and quality issues recorded from execution.

2\. Preprocessing fit on training data only; SMOTE (if used) inside CV pipeline.

3\. Model selection by cross-validation on training data; test set used once.

4\. Final model justified by recall and PR-AUC, not accuracy alone.

5\. Every README claim traces to a saved output.

6\. Honest limitations: association not causation, benchmark data, prototype.



\## Evidence rule

No metric, statistic or finding is reported unless produced by executed code.

Any reference presentation's numbers are treated as unverified.

