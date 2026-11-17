# Subgroup Evaluation

## Purpose

This phase evaluates the frozen Logistic Regression model across predefined test-set subgroups.

The purpose is to assess whether model performance differs across gender and age groups. This is an **evaluation-only** analysis: no model tuning, threshold selection, feature modification, or retraining decisions are made from these subgroup results.

The held-out test set remains separate from model development and selection.

## Evaluation Setup

- **Model:** Frozen Logistic Regression
- **Classification threshold:** `0.50`
- **Test rows:** 294
- **Test attrition cases:** 47
- **Threshold selection:** Fixed before subgroup evaluation
- **Model modification:** None
- **Purpose:** Evaluation only

The same preprocessing and Logistic Regression specification used for the final evaluation were applied.

## Overall Test-Set Performance

| Metric | Overall |
|---|---:|
| Rows | 294 |
| Attrition cases | 47 |
| Attrition rate | 15.99% |
| Precision | 34.88% |
| Recall | 63.83% |
| False-positive rate | 22.67% |
| Flag rate | 29.25% |

The overall confusion matrix was:

| | Predicted No Attrition | Predicted Attrition |
|---|---:|---:|
| Actual No Attrition | 191 | 56 |
| Actual Attrition | 17 | 30 |

## Gender Subgroups

| Gender | Rows | Attrition Cases | Attrition Rate | Precision | Recall | False-Positive Rate | Flag Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Female | 116 | 16 | 13.79% | 32.35% | 68.75% | 23.00% | 29.31% |
| Male | 178 | 31 | 17.42% | 36.54% | 61.29% | 22.45% | 29.21% |

Recall is higher for the Female subgroup (68.75%) than for the Male subgroup (61.29%). Precision is somewhat higher for the Male subgroup.

The false-positive rates are relatively close: 23.00% for Female employees and 22.45% for Male employees.

The subgroup results therefore show some performance variation, but the differences should not be interpreted as evidence of model bias by themselves. Sample sizes and the underlying attrition rates also differ between the groups.

## Age Subgroups

The test set was divided into four predefined age groups.

| Age Group | Rows | Attrition Cases | Attrition Rate | Precision | Recall | False-Positive Rate | Flag Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 18–30 | 77 | 18 | 23.38% | 43.75% | 77.78% | 30.51% | 41.56% |
| 31–36 | 84 | 11 | 13.10% | 25.81% | 72.73% | 31.51% | 36.90% |
| 37–43 | 72 | 7 | 9.72% | 35.71% | 71.43% | 13.85% | 19.44% |
| 44+ | 61 | 11 | 18.03% | 33.33% | 27.27% | 12.00% | 14.75% |

The largest difference appears in recall.

The model recalls 77.78% of attrition cases in the 18–30 group, compared with 27.27% in the 44+ group.

The 44+ subgroup also has the lowest flag rate at 14.75%, while the 18–30 subgroup has the highest at 41.56%.

These differences indicate that model performance is not uniform across age groups.

## Interpretation

The subgroup analysis highlights an important distinction between **overall model performance** and **performance within individual groups**.

For gender, recall and false-positive rates are relatively similar, although recall is higher for Female employees.

For age, the differences are substantially larger. The model identifies a much smaller proportion of observed attrition cases in the 44+ subgroup than in the younger groups.

The age-group differences should be investigated further before considering any real-world deployment. However, this analysis alone does not establish that the model is unfair or discriminatory.

Possible explanations include:

- different attrition rates across age groups,
- differences in feature distributions,
- limited numbers of attrition cases within some subgroups,
- relationships between age and other predictors,
- and differences in how well the model separates attrition from non-attrition within each subgroup.

## Bootstrap uncertainty

The generated subgroup table includes percentile-bootstrap 95% intervals using 2,000 resamples per subgroup. Intervals are reported for both observed attrition rate and threshold-based recall. They quantify uncertainty around this held-out sample; they are not fairness certification and should not be interpreted as population guarantees.

For the **44+** group, the model recalls **3 of 11 attrition cases (27.27%)**. The small numerator and denominator make this estimate particularly unstable, so the confidence interval should be read alongside the point estimate rather than treated as a precise group-level performance claim.

## Important Limitations

This analysis has several limitations.

1. **Small subgroup sample sizes:** Some subgroups contain relatively few attrition cases. For example, the 37–43 group contains only 7 attrition cases and the 44+ group contains 11.

2. **Limited statistical inference:** Bootstrap intervals are now reported for attrition rate and recall. They quantify sampling uncertainty but do not establish fairness or causality.

3. **Single held-out split:** Results depend on the particular test partition and may vary on another sample.

4. **Benchmark dataset:** The IBM HR Analytics dataset is a fictional benchmark dataset and does not represent an actual organization's workforce.

5. **No model changes:** Subgroup results were not used to tune the model, modify the threshold, or change features.

6. **Fairness cannot be concluded from these metrics alone:** A complete fairness assessment would require a predefined fairness framework, appropriate statistical uncertainty analysis, and consideration of the intended use and organizational context.

## Responsible Use

Subgroup evaluation should be used as a diagnostic and monitoring step rather than as a basis for treating employees differently.

In a real employment setting, predicted attrition risk should not be used as the sole basis for employment decisions.

Any practical deployment would require:

- privacy safeguards,
- access controls,
- fairness evaluation,
- uncertainty analysis,
- organization-specific validation,
- monitoring for model drift,
- human oversight,
- and explicit governance for employment-related use.

## Test-Set Separation

The subgroup evaluation uses the held-out test partition only for evaluation.

The subgroup results were not used to select the Logistic Regression model, tune its hyperparameters, choose the classification threshold, or modify the final model.

The final overall test evaluation remains documented separately in:

`docs/final_evaluation.md`

The model interpretation analysis remains documented separately in:

`docs/model_interpretation.md`

## Generated Artifact

The subgroup evaluation results are saved in:

`reports/subgroup_evaluation.csv`

The source script is:

`src/subgroup_evaluation.py`
