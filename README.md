# Employee Attrition Analysis

[![CI](https://github.com/pranjali-sonkamble/employee-attrition-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/pranjali-sonkamble/employee-attrition-analysis/actions/workflows/ci.yml)

An end-to-end analytical prototype for understanding employee attrition patterns and evaluating a leakage-safe machine-learning **attrition-ranking** model on the IBM HR Analytics Employee Attrition & Performance benchmark dataset.

> **Not a production HR decision system.** The dataset is fictional, and the model must not be used as the sole basis for employment decisions.

## Project at a glance

| Item | Result |
|---|---:|
| Dataset | IBM HR Analytics Employee Attrition & Performance |
| Dataset shape | 1,470 × 35 |
| Modeling features | 30 |
| Train / test | 1,176 / 294 |
| Selected model | Logistic Regression |
| Test Average Precision | **0.5612** |
| Test ROC-AUC | **0.8032** |
| Test precision / recall | **0.3488 / 0.6383** |
| Test F1 | **0.4511** |
| Fixed threshold | 0.50 |
| Bootstrap resamples | 2,000 |

### What this means

At the fixed 0.50 threshold, the model catches **30 of 47** observed attrition cases while flagging **86 of 294** test records. If an organization could review only the **top 20% of scored employees**, the frozen test predictions identify about **53% of observed leavers (95% bootstrap CI: 42%–68%)**.

This is a screening/ranking result, not a calibrated probability of an employee leaving.

## Business findings

The benchmark contains several strong descriptive associations:

- **Overtime:** 30.53% observed attrition vs 10.44% without overtime.
- **Sales Representatives:** 39.76% observed attrition.
- **Early tenure (≤3 years):** 25.96% observed attrition.
- **Single employees:** 25.53% observed attrition.

These are associations in a fictional benchmark, not causal effects. They should motivate investigation rather than automatic HR action.

## Analytical workflow

**Data validation → EDA → leakage-safe preprocessing → repeated cross-validation → OOF threshold analysis → fixed model/threshold → held-out test evaluation → bootstrap uncertainty → model interpretation → subgroup diagnostics → sensitive-attribute ablation → regression tests**

Preprocessing is inside the modeling pipeline: numeric imputation/scaling and categorical imputation/one-hot encoding are fitted from training data only. The primary Logistic Regression uses `class_weight="balanced"`. **SMOTE is disabled** because class weighting addresses the class imbalance without synthetic observations and keeps the final pipeline simpler; SMOTE was not separately benchmarked.

Model comparison uses repeated stratified 5-fold CV on the training partition:

| Model | CV AP | CV ROC-AUC |
|---|---:|---:|
| Logistic Regression | **0.6130 ± 0.0065** | **0.8251 ± 0.0017** |
| Hist Gradient Boosting | 0.5472 ± 0.0162 | 0.8005 ± 0.0098 |
| Random Forest | 0.5460 ± 0.0091 | 0.8026 ± 0.0001 |
| Dummy baseline | 0.1616 ± 0.0000 | 0.5000 ± 0.0000 |

Logistic Regression is therefore the strongest of the **fixed configurations tested**, not proof that it is globally optimal.

## Threshold and final evaluation

Thresholds were evaluated using training-set out-of-fold predictions only. The final **0.50** threshold was fixed before held-out evaluation rather than selected by optimizing the test set. No organization-specific cost ratio or review-capacity constraint was predefined, so 0.50 is treated as a reference operating point, not an optimal business threshold.

| Method | Precision | Recall | F1 | Flag rate |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.3488 | **0.6383** | **0.4511** | 29.25% |
| Overtime + short-tenure rule | **0.5484** | 0.3617 | 0.4359 | **10.54%** |

The model and rule have **comparable F1 (0.451 vs 0.436)** on only 47 positive test cases; this should not be described as a decisive ML win. The rule has higher precision and lower workload, while Logistic Regression has substantially higher recall. Because the rule cutoffs were motivated by full-dataset EDA, it is an **illustrative, post-hoc baseline**. AP/ROC-AUC for the binary rule score are not directly comparable to the model's ranking metrics.

## Responsible AI and calibration

Age, Gender and MaritalStatus remain in the primary benchmark-schema model. A controlled ablation removes all three while keeping the split, model and threshold unchanged:

| Metric | Full | Without Age/Gender/MaritalStatus |
|---|---:|---:|
| Average Precision | 0.5612 | 0.5433 |
| ROC-AUC | 0.8032 | 0.7801 |
| Precision | 0.3488 | 0.3659 |
| Recall | 0.6383 | 0.6383 |

The ablation is exploratory and was not used to rewrite the primary specification. Removing these fields also does not eliminate proxy concerns.

The held-out Brier score is **0.1561**, versus **0.1343** for a constant prediction equal to the test-set base rate. Because `class_weight="balanced"` can distort raw probabilities, the predicted probabilities are **not treated as calibrated risk estimates**; they are used as ranking scores.

Subgroup diagnostics are descriptive. The 44+ subgroup recalls **3 of 11** attrition cases (27.27%; bootstrap 95% CI 0%–55.57%), so its estimate is highly uncertain.

## Limitations

- The IBM HR dataset is a fictional benchmark and is not evidence about a real employer.
- The held-out test contains only 47 positive cases.
- The test set was reused for several evaluation-only diagnostics after the model and threshold were frozen; results are disclosed rather than treated as independent validation.
- The rule baseline is post-hoc.
- The model is not calibrated and has substantial false positives at the fixed threshold.
- Associations do not establish causation.
- A real deployment would require time-based validation, calibration, drift monitoring, predefined fairness criteria, privacy/access controls, governance, consent, and human review.

## Reproducibility

Python 3.10 is supported. Install the pinned environment:

```bash
pip install -r requirements-lock.txt
```

Run tests:

```bash
pytest -q
```

Run the full analysis:

```bash
python src/run_all.py
```

Or run individual stages from the repository root:

```bash
python src/inspect_data.py
python src/validate_columns.py
python src/explore_data.py
python src/explore_relationships.py
python src/compare_models.py
python src/evaluate_thresholds.py
python src/final_evaluation.py
python src/test_uncertainty.py
python src/model_interpretation.py
python src/subgroup_evaluation.py
python src/sensitive_attribute_ablation.py
python src/rule_baseline.py
python src/segment_statistics.py
python src/calibration.py
```

The raw CSV is intentionally not committed. See [data/README.md](data/README.md) and [docs/data_source.md](docs/data_source.md).

## Repository structure

```text
employee-attrition-analysis/
├── data/README.md
├── docs/
│   ├── project_summary.md
│   ├── results_workbook.md
│   ├── calibration.md
│   ├── data_source.md
│   ├── eda_findings.md
│   ├── final_evaluation.md
│   ├── main_segment_statistics.md
│   ├── model_interpretation.md
│   ├── project_charter.md
│   ├── rule_baseline.md
│   ├── subgroup_evaluation.md
│   ├── threshold_evaluation.md
│   └── sensitive_attribute_ablation.md
├── .github/workflows/ci.yml
├── reports/
│   ├── figures/
│   ├── main_segment_statistics.csv
│   ├── rule_baseline_comparison.csv
│   ├── calibration_metrics.csv
│   ├── final_test_metrics.csv
│   ├── final_test_confusion_matrix.csv
│   ├── sensitive_attribute_ablation.csv
│   ├── sensitive_attribute_ablation_subgroups.csv
│   ├── subgroup_evaluation.csv
│   ├── test_uncertainty.csv
│   └── ...
├── src/
│   ├── run_all.py
│   ├── rule_baseline.py
│   ├── segment_statistics.py
│   ├── calibration.py
│   └── ...
├── tests/
│   ├── test_data_pipeline.py
│   └── test_behavior.py
├── dashboard/
│   └── Employee_Attrition_Recruiter_Dashboard.xlsx
├── requirements.txt
├── requirements-lock.txt
├── LICENSE
└── .gitignore
```

The Excel file is a **static results workbook**, not an interactive BI dashboard. The Python pipeline and CSV outputs remain the analytical source of truth.
