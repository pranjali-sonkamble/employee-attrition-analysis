\# Decision-Threshold Evaluation



\## Objective



Evaluate how changing the classification threshold affects precision, recall, F1 score, and the number of employees flagged for potential attrition. The objective is to understand the operational trade-off rather than assume the default threshold is optimal.



\## Methodology



\- Dataset: IBM HR Analytics Employee Attrition dataset.

\- Training partition: 1,176 employees, including 190 attrition cases.

\- Validation approach: five-fold stratified cross-validation with out-of-fold predicted probabilities.

\- Models evaluated: Logistic Regression and Random Forest.

\- Thresholds evaluated: 0.10 to 0.60 in increments of 0.05.

\- Preprocessing: the existing pipeline, fitted separately within each cross-validation fold.

\- Resampling: SMOTE disabled.

\- Held-out test partition: not used.



\## Results



\### Logistic Regression



\- At threshold 0.10, recall was 94.21%, precision was 20.84%, and 859 employees were flagged.

\- At threshold 0.50, recall was 73.68%, precision was 36.84%, and 380 employees were flagged.

\- At threshold 0.60, recall was 66.84%, precision was 45.20%, and 281 employees were flagged.

\- The highest F1 score among the tested thresholds was 0.539 at threshold 0.60.



\### Random Forest



\- At threshold 0.20, recall was 75.26%, precision was 35.40%, and 404 employees were flagged.

\- At threshold 0.25, recall was 63.16%, precision was 43.80%, and 274 employees were flagged.

\- At threshold 0.30, recall was 52.11%, precision was 50.25%, and 197 employees were flagged.

\- The highest F1 score among the tested thresholds was 0.517 at threshold 0.25.



\## Interpretation



Lower thresholds generally increase recall and the number of employees flagged. This can reduce missed attrition cases while increasing false positives and the effort required to review cases.



Higher thresholds can improve precision and reduce the number of flagged employees, but they also increase the number of missed attrition cases.



The appropriate operating threshold depends on the organization's retention objectives, review capacity, and the relative costs of false positives and false negatives. These results alone do not establish a universally optimal threshold.



\## Limitations and safeguards



\- Threshold performance is estimated from training-set out-of-fold predictions and is not final test-set performance.

\- The threshold grid is discrete; the best observed threshold is not necessarily the global optimum.

\- Cross-validation fold predictions are combined to calculate the displayed metrics. These are useful development estimates, but they do not replace final evaluation on untouched data.

\- Attrition associations are not proof of causation or reliable predictions of individual intent.

\- Model outputs should not be used as the sole basis for employment decisions. Any proposed retention use requires privacy, fairness, and appropriate human oversight.



\## Next steps



1\. Define an explicit operational objective and acceptable false-positive burden before selecting a threshold.

2\. Record the threshold-selection rationale without using the held-out test partition.

3\. Complete model selection and document any tuning decisions.

4\. Evaluate the chosen model and threshold on the untouched test partition once, during the final evaluation phase.



