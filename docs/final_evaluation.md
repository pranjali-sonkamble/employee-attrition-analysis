# Final Model Evaluation

## Selected Model

Logistic Regression was selected before the final held-out test evaluation, based on repeated stratified five-fold cross-validation across three repeats. The regenerated model-comparison results show mean average precision of 0.6130 and mean ROC-AUC of 0.8251, outperforming the other evaluated classification models on these metrics.

The model and threshold were fixed before the final test evaluation. No hyperparameter tuning was performed as part of the final test evaluation.

## Evaluation Method

- **Dataset:** IBM HR Analytics Employee Attrition dataset
- **Training partition:** 1,176 employees
- **Held-out test partition:** 294 employees, including 47 attrition cases
- **Model:** Logistic Regression, with `C=1.0` and `class_weight="balanced"`
- **Preprocessing:** Median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features
- **Resampling:** SMOTE disabled
- **Classification threshold:** 0.50
- **Evaluation:** The model was fitted on the training partition and evaluated on the held-out test partition.

The test results below are reported for the existing fixed model and threshold. The held-out test set must not be used for further model selection, feature changes, or threshold tuning.

## Final Test Results

| Metric | Result |
|---|---:|
| Average precision | 0.5612 |
| ROC-AUC | 0.8032 |
| Precision | 0.3488 |
| Recall | 0.6383 |
| F1 score | 0.4511 |

## Confusion Matrix

| Actual outcome | Predicted no attrition | Predicted attrition |
|---|---:|---:|
| No attrition | 191 | 56 |
| Attrition | 17 | 30 |

At the fixed threshold of 0.50, the model identified 30 of the 47 employees who left and missed 17. It flagged 86 employees in total, of whom 56 did not leave.

This corresponds to a flag rate of approximately 29% of the test population. Among the employees flagged, approximately 35% actually left. In a practical HR workflow, false positives can create unnecessary follow-up conversations and consume limited retention resources.

Recall is approximately 64% on this test split. A Wilson 95% confidence interval for 30 successes among 47 attrition cases is approximately 50%–76%. This interval illustrates the uncertainty associated with the limited number of test-set attrition cases; it does not guarantee performance on other populations.

## Cross-Validation and Test Comparison

| Metric | Repeated-CV mean | Final test |
|---|---:|---:|
| Average precision | 0.6130 | 0.5612 |
| ROC-AUC | 0.8251 | 0.8032 |
| Recall at threshold 0.50 | 0.737 | 0.638 |

The test metrics are lower than the cross-validation estimates. The recall difference is consistent with sampling variability given the limited number of attrition cases and should not, by itself, be interpreted as proof of overfitting.

For HR planning, a complementary training-set ranking view is useful: the top 20% of employees ranked by Logistic Regression's out-of-fold scores captured approximately 63% of leavers, with approximately 51% precision. This is a training-set cross-validation result, not a result from the held-out test set.

## Limitations and Responsible Use

- Results are based on one held-out split and may not generalize to other organizations or future employee populations.
- The threshold of 0.50 is a fixed reference threshold, not a universally optimal business decision threshold.
- No hyperparameter tuning was performed as part of this final evaluation. The Logistic Regression settings were fixed at `C=1.0` and `class_weight="balanced"`.
- Attrition predictions identify statistical patterns and do not establish why an employee might leave.
- Fairness checks and permutation-importance analysis should use training-set out-of-fold predictions rather than small test subgroups.
- Practical use requires privacy safeguards, fairness evaluation, and human oversight. Predictions must not be the sole basis for employment decisions.

## Generated Artifacts

- `reports/final_test_metrics.csv`
- `reports/final_test_confusion_matrix.csv`
- `reports/figures/final_test_confusion_matrix.png`
- `reports/model_comparison.csv`
- `reports/model_comparison_repeats.csv`
- `reports/figures/pr_curves_oof.png`
