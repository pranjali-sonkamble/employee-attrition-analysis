# Project Summary

This repository is an analytical prototype for employee attrition ranking on the IBM HR Analytics benchmark.

## Headline result

The primary benchmark-schema Logistic Regression achieves **0.5612 Average Precision**, **0.8032 ROC-AUC**, **0.3488 precision**, **0.6383 recall**, and **0.4511 F1** on the untouched 294-row test partition at the fixed 0.50 threshold.

At a **top-20% review capacity**, the frozen test ranking identifies about **53% of observed leavers (95% bootstrap CI: 42%–68%)**.

## Responsible-use result

The model uses `class_weight="balanced"`. Its held-out Brier score (**0.1561**) is worse than the constant observed-base-rate reference (**0.1343**), so the raw predicted probabilities are **not treated as calibrated risk estimates**. They are used as ranking/screening scores.

Age, Gender and MaritalStatus remain in the primary benchmark-schema model. A controlled ablation removes all three without changing the split, model specification or threshold. The ablation is exploratory and does not justify changing the primary specification after seeing test results.

## Rule baseline

The illustrative rule `OverTime == "Yes" AND YearsAtCompany <= 3` has **0.4359 F1**, compared with **0.4511** for Logistic Regression, on the same test rows. The F1 difference should be treated as a tie/comparable result given the 47 positive test cases. The rule has higher precision and lower flag rate; the model has higher recall.

## Scope and limitations

- The dataset is fictional and does not establish real-world employee behavior.
- Observational associations are not causal.
- The held-out test contains only 47 positive cases.
- The test set is reused for several frozen-model diagnostics, which are disclosed as evaluation-only analyses rather than independent validation.
- The model is not calibrated and should not be used as the sole basis for employment decisions.

See the linked calibration, final evaluation, subgroup, and sensitive-attribute-ablation documents for methodology and evidence.
