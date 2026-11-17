# Decision-Threshold Evaluation

## Objective

Evaluate how the classification threshold changes precision, recall, F1 score, and the number of records flagged for potential attrition. The purpose is to understand the trade-off, not to claim that one threshold is universally optimal.

## Methodology

- **Dataset:** IBM HR Analytics Employee Attrition benchmark.
- **Training partition:** 1,176 employees, including 190 attrition cases.
- **Validation:** Five-fold stratified cross-validation with out-of-fold (OOF) predicted probabilities.
- **Models evaluated:** Logistic Regression and Random Forest.
- **Threshold grid:** 0.10 to 0.60 in increments of 0.05.
- **Preprocessing:** Fitted independently within each CV fold through the modeling pipeline.
- **SMOTE:** Disabled.
- **Held-out test partition:** Not used for threshold analysis.

The figures in this document are development estimates calculated from the training partition's OOF predictions. They are not final test-set results.

## Selected results

### Logistic Regression

| Threshold | Precision | Recall | F1 | Flagged training rows |
|---:|---:|---:|---:|---:|
| 0.10 | 20.84% | 94.21% | 0.341 | 859 |
| 0.50 | 36.84% | 73.68% | 0.491 | 380 |
| 0.60 | 45.20% | 66.84% | 0.539 | 281 |

The highest F1 among the tested Logistic Regression thresholds was 0.539 at 0.60.

### Random Forest

| Threshold | Precision | Recall | F1 | Flagged training rows |
|---:|---:|---:|---:|---:|
| 0.20 | 35.40% | 75.26% | 0.481 | 404 |
| 0.25 | 43.80% | 63.16% | 0.517 | 274 |
| 0.30 | 50.25% | 52.11% | 0.512 | 197 |

The highest F1 among the tested Random Forest thresholds was 0.517 at 0.25.

Full results are in [reports/threshold_evaluation.csv](../reports/threshold_evaluation.csv).

## Interpretation

Lower thresholds generally increase recall and the number of records flagged. This can reduce missed attrition cases while increasing false positives and review workload.

Higher thresholds generally improve precision and reduce the number of flagged records, while missing more attrition cases.

The appropriate threshold depends on the intended workflow, review capacity, and the relative costs of false positives and false negatives. The benchmark results do not establish a universally optimal operating threshold.

## Relationship to the final evaluation

The selected model was compared using repeated stratified five-fold CV in [reports/model_comparison.csv](../reports/model_comparison.csv). Logistic Regression was selected on the training partition before the held-out test evaluation.

The final evaluation used a fixed reference threshold of **0.50** and the untouched 294-row test partition. The OOF analysis showed that 0.60 increased F1 from 0.491 to 0.539 but reduced recall from 73.68% to 66.84%. Because no organization-specific cost function or review-capacity constraint was predefined, 0.50 was retained as the fixed reference threshold rather than selecting 0.60 solely to maximize F1. At 0.50, the final test evaluation flags 86 of 294 records, a **29.25% flag rate (about 29%)**. The reported test metrics therefore remain associated with the fixed 0.50 reference threshold.

## Limitations and safeguards

- The threshold grid is discrete; the best tested threshold is not necessarily the global optimum.
- OOF results are development estimates and do not replace an untouched test evaluation.
- Cross-validation fold predictions are combined to calculate the displayed metrics.
- The benchmark is fictional and cannot establish how a real organization's employees behave.
- Predictions are statistical signals, not evidence of individual intent or causal explanations.
- Model outputs must not be the sole basis for employment decisions. Any real-world use would require organization-specific validation, privacy safeguards, fairness evaluation, and human oversight.

## Reproduction

Run from the repository root after placing the raw CSV at the location described in [data/README.md](../data/README.md):

```bash
python src/evaluate_thresholds.py
```
