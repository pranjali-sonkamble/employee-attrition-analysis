# Simple Rules Baseline

## Purpose

The project now includes a deliberately simple, interpretable baseline:

**flag = OverTime == "Yes" AND YearsAtCompany <= 3**

The rule is an illustrative, post-hoc baseline and requires no model fitting. It is evaluated on the same held-out 294-row test partition as the frozen Logistic Regression.

## Why it matters

A machine-learning model should add value beyond an obvious business-rule heuristic. Comparing both methods on the same test rows makes that question explicit.

The comparison includes Average Precision, ROC-AUC, Precision, Recall, F1, flag rate, and confusion-matrix counts.

## Reproduction

    python src/rule_baseline.py

Output: reports/rule_baseline_comparison.csv

A stronger Logistic Regression result means the ML model provides incremental predictive discrimination over this simple rule in this benchmark. It does not mean the model explains attrition or that either method should be used as an employment decision rule.
