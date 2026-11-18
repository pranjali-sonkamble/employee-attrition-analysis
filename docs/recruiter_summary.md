# Recruiter summary

This repository is the source of truth for the Employee Attrition Analysis project. The recruiter-facing summary is maintained here in Markdown so the link remains valid when the repository is cloned.

## Project focus

Predict employee attrition on the public IBM HR Analytics benchmark while demonstrating leakage-aware preprocessing, model comparison, held-out evaluation, calibration diagnostics, a simple rule baseline, subgroup analysis, and sensitive-attribute ablation.

## Headline result

The primary benchmark-schema Logistic Regression achieves **0.5612 Average Precision**, **0.8032 ROC-AUC**, **0.3488 precision**, **0.6383 recall**, and **0.4511 F1** on the untouched 294-row test partition at the fixed 0.50 threshold.

## Responsible-use result

The model uses `class_weight="balanced"`. Its held-out Brier score (**0.1561**) is worse than the constant observed-base-rate reference (**0.1343**), so the raw predicted probabilities are **not treated as calibrated risk estimates**. The project uses them as ranking/screening scores.

Age, Gender, and MaritalStatus are evaluated in a controlled ablation. The ablation is exploratory; it is not presented as a post-hoc reason to change the primary benchmark specification.

See [calibration](calibration.md), [sensitive-attribute ablation](sensitive_attribute_ablation.md), and [final evaluation](final_evaluation.md) for evidence and limitations.

The Excel recruiter dashboard is available at [dashboard/Employee_Attrition_Recruiter_Dashboard.xlsx](../dashboard/Employee_Attrition_Recruiter_Dashboard.xlsx).
