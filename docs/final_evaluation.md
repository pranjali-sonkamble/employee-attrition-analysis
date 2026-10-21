\# Final Model Evaluation



\## Selected Model



Logistic Regression was selected as the candidate model based on repeated stratified five-fold cross-validation across three repeats. It achieved mean average precision of 0.6130 and mean ROC-AUC of 0.8251, outperforming the other evaluated classification models on these metrics.



\## Evaluation Method



\- Dataset: IBM HR Analytics Employee Attrition dataset

\- Training partition: 1,176 employees

\- Held-out test partition: 294 employees, including 47 attrition cases

\- Preprocessing: median imputation and standard scaling for numeric and ordinal features; most-frequent imputation and one-hot encoding for categorical features

\- Resampling: SMOTE disabled

\- Classification threshold: 0.50

\- The model was fitted on the training partition and evaluated on the held-out test partition.



\## Final Test Results



| Metric | Result |

|---|---:|

| Average precision | 0.5612 |

| ROC-AUC | 0.8032 |

| Precision | 0.3488 |

| Recall | 0.6383 |

| F1 score | 0.4511 |



\## Confusion Matrix



| Actual outcome | Predicted no attrition | Predicted attrition |

|---|---:|---:|

| No attrition | 191 | 56 |

| Attrition | 17 | 30 |



At the fixed threshold of 0.50, the model identified 30 of the 47 employees who left. It missed 17 attrition cases and flagged 56 employees who did not leave.



\## Cross-Validation and Test Comparison



| Metric | Repeated-CV mean | Final test |

|---|---:|---:|

| Average precision | 0.6130 | 0.5612 |

| ROC-AUC | 0.8251 | 0.8032 |



The lower test metrics indicate some reduction in performance on the held-out partition. This difference should be reported transparently rather than concealed by further tuning on the test set.



\## Limitations and Responsible Use



\- These results come from one held-out split and may not generalize to other organizations or future employee populations.

\- The threshold of 0.50 is a fixed reference threshold, not a universally optimal business decision threshold.

\- Attrition predictions identify statistical patterns and do not establish why an employee might leave.

\- Any practical use requires privacy safeguards, fairness evaluation, and human oversight. Predictions must not be the sole basis for employment decisions.



\## Generated Artifacts



\- `reports/final\_test\_metrics.csv`

\- `reports/final\_test\_confusion\_matrix.csv`

\- `reports/figures/final\_test\_confusion\_matrix.png`



