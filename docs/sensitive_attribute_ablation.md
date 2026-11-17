# Sensitive-Attribute Ablation

## Purpose

This analysis compares the frozen Logistic Regression specification with an otherwise identical model that excludes **Age**, **Gender**, and **MaritalStatus** from model inputs.

The same stratified train/test split, preprocessing approach, classifier configuration, random seed, and fixed 0.50 threshold are used for both models. The three attributes remain available only for subgroup grouping in the ablated evaluation.

## Method

1. Load the same 1,470-row benchmark dataset and reproduce the existing stratified 80/20 split.
2. Fit the original Logistic Regression pipeline using all 30 modeling features.
3. Fit a second pipeline after removing Age, Gender, and MaritalStatus from the modeling matrix.
4. Evaluate both models on the identical untouched 294-row test partition.
5. Compare Average Precision, ROC-AUC, precision, recall, confusion-matrix counts, and flag rate.
6. Compare subgroup recall, precision, false-positive rate, and flag rate for Gender, MaritalStatus, and the predefined age bands.

The ablation does not change the threshold or tune the model after seeing the test results.

## Reproduction

From the repository root, after placing the raw CSV at the path documented in data/README.md:

python src/sensitive_attribute_ablation.py

The script writes:

- reports/sensitive_attribute_ablation.csv
- reports/sensitive_attribute_ablation_subgroups.csv

and regenerates this document with the current numeric results.

## Interpretation standard

A difference between the full and ablated models indicates that the excluded attributes contribute predictive information in this benchmark. It does **not** establish causality, discrimination, fairness, or what would happen in another organization.

The subgroup analysis is descriptive. Because the test set contains only 47 attrition cases, small subgroup differences should not be over-interpreted.

## Recorded results

| Metric | Full model | Without Age/Gender/MaritalStatus |
|---|---:|---:|
| Average Precision | 0.5612 | 0.5433 |
| ROC-AUC | 0.8032 | 0.7801 |
| Precision | 0.3488 | 0.3659 |
| Recall | 0.6383 | 0.6383 |
| Flag rate | 29.25% | 27.89% |
| False positives | 56 | 52 |

At the fixed 0.50 threshold, recall is unchanged at 63.83%, while Average Precision decreases by 0.0179 and ROC-AUC decreases by 0.0231 after removing the three attributes. Precision increases by 0.0171 and the number of false positives falls from 56 to 52.

Gender recall is unchanged for Female (68.75%) and Male (61.29%). Age-group recall changes from 77.78% to 72.22% for ages 18–30, 72.73% to 63.64% for ages 31–36, 71.43% to 85.71% for ages 37–43, and 27.27% to 36.36% for ages 44+. These subgroup differences are based on a single held-out split with limited attrition counts and should therefore be treated as diagnostic rather than evidence of production-level fairness.

## Responsible use

The ablation is an analytical diagnostic, not evidence that either feature set is appropriate for employment decisions. Any real HR application would require organization-specific validation, privacy safeguards, a predefined fairness framework, governance, monitoring, and meaningful human oversight.

> **Numeric results are generated directly from the project's raw dataset by src/sensitive_attribute_ablation.py; they are not manually entered or estimated.**
