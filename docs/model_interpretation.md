# Model Interpretation

## Purpose

This phase interprets the frozen Logistic Regression model selected during repeated cross-validation. Interpretation is performed using the training partition only. The held-out test set is not used to calculate coefficients, feature rankings, or explanatory statistics.

## Model Used

- **Model:** Logistic Regression
- **Training rows:** 1,176
- **Transformed features:** 51
- **Regularization:** `C=1.0`
- **Class weighting:** `balanced`
- **SMOTE:** disabled for the final frozen model
- **Interpretation data:** training partition only

The model is implemented as a pipeline containing preprocessing followed by Logistic Regression. This ensures that the preprocessing used for interpretation is the same preprocessing used during model evaluation.

## How to Read the Coefficients

For Logistic Regression, each coefficient represents the change in the **log-odds of attrition** associated with a one-unit increase in the corresponding transformed feature, holding the other model features constant.

For a coefficient `β`, the corresponding odds ratio is:

`odds ratio = exp(β)`

- Positive coefficient → higher modelled odds of attrition.
- Negative coefficient → lower modelled odds of attrition.
- Odds ratio greater than 1 → a positive contribution to the modelled log-odds when that transformed feature is active.
- Odds ratio less than 1 → a negative contribution to the modelled log-odds when that transformed feature is active.

The categorical encoder uses `drop=None`, so **all observed categories are retained**; no category is omitted as a reference level. Therefore, individual categorical coefficients should not be described as simple category-versus-reference effects. They are conditional coefficients in the regularized design matrix and should be interpreted jointly with the intercept and other encoded features.

**Important:** Numeric variables were standardized during preprocessing. Therefore, an odds ratio for a numeric feature represents the change in modelled odds associated with a **one-standard-deviation increase in the original variable**, not necessarily a one-unit increase in the original measurement.

For example, the odds ratio for `TotalWorkingYears` does **not** mean that one additional year of experience multiplies the odds by 0.54. It represents the modelled change associated with a one-standardized-unit increase in `TotalWorkingYears`.

These coefficients should be interpreted as model associations, not causal effects.

## Top Positive Coefficients

| Feature | Coefficient | Odds Ratio |
|---|---:|---:|
| JobRole_Laboratory Technician | 1.1943 | 3.3014 |
| JobRole_Sales Representative | 1.1206 | 3.0667 |
| BusinessTravel_Travel Frequently | 0.7876 | 2.1980 |
| OverTime_Yes | 0.7615 | 2.1415 |
| EducationField_Human Resources | 0.7211 | 2.0568 |
| JobLevel | 0.6074 | 1.8357 |
| YearsSinceLastPromotion | 0.5025 | 1.6529 |
| JobRole_Human Resources | 0.4639 | 1.5902 |
| NumCompaniesWorked | 0.4605 | 1.5849 |
| MaritalStatus_Single | 0.4573 | 1.5798 |

### Interpretation

The largest positive coefficient belongs to the transformed `JobRole_Laboratory Technician` feature. Its coefficient is approximately 1.19, corresponding to an odds ratio of approximately 3.30 for that transformed feature when active, conditional on the other variables in the model.

`JobRole_Sales Representative` is also strongly positive, with an odds ratio of approximately 3.07 for that transformed feature when active.

`BusinessTravel_Travel Frequently` and `OverTime_Yes` are also among the strongest positive associations. The `OverTime_Yes` coefficient corresponds to approximately 2.14 times the modelled odds contribution associated with that transformed feature when active, holding the other model features constant.

The positive coefficient for `JobLevel` indicates a positive modelled association after accounting for the other included features. Because `JobLevel` is a standardized numeric feature and is correlated with several experience and compensation variables, its odds ratio should not be interpreted as the effect of moving up exactly one job level.

Similarly, the positive coefficients for `YearsSinceLastPromotion` and `NumCompaniesWorked` represent associations per one standardized unit of those variables.

## Top Negative Coefficients

| Feature | Coefficient | Odds Ratio |
|---|---:|---:|
| JobRole_Research Director | -1.4657 | 0.2309 |
| BusinessTravel_Non-Travel | -0.9939 | 0.3701 |
| OverTime_No | -0.8928 | 0.4095 |
| EducationField_Other | -0.7981 | 0.4502 |
| JobRole_Healthcare Representative | -0.6472 | 0.5235 |
| TotalWorkingYears | -0.6202 | 0.5378 |
| MaritalStatus_Divorced | -0.4822 | 0.6174 |
| YearsWithCurrManager | -0.4547 | 0.6347 |
| EnvironmentSatisfaction | -0.4262 | 0.6530 |
| JobSatisfaction | -0.3889 | 0.6778 |

### Interpretation

`JobRole_Research Director` has the most negative coefficient, with an odds ratio of approximately 0.23 for that transformed feature when active.

`BusinessTravel_Non-Travel` and `OverTime_No` are also strongly negative. The `OverTime_No` coefficient is the expected counterpart to the positive association for `OverTime_Yes`, although the two coefficients should not be interpreted independently of the categorical encoding and other model features.

Several experience and satisfaction variables have negative coefficients.

For example, `TotalWorkingYears` has an odds ratio of approximately 0.54. Because `TotalWorkingYears` was standardized before fitting the model, this represents the modelled change in odds associated with a one-standard-deviation increase in total working years, conditional on the other included features.

The same interpretation principle applies to `YearsWithCurrManager`, `EnvironmentSatisfaction`, and `JobSatisfaction`: their odds ratios describe changes per standardized unit rather than per raw unit of measurement.

## Relationship to EDA

The coefficient results should be considered alongside the earlier exploratory analysis rather than treated as an independent causal ranking.

The EDA showed substantially different observed attrition rates across several categorical groups. For example:

- Employees working overtime had an observed attrition rate of 30.53%, compared with 10.44% for employees without overtime.
- Sales Representatives had an observed attrition rate of 39.76%.
- Laboratory Technicians had an observed attrition rate of 23.94%.
- Research Directors had an observed attrition rate of 2.50%.
- Single employees had an observed attrition rate of 25.53%.

These descriptive rates provide context for why several of the corresponding categorical variables receive strong coefficients in the Logistic Regression model.

However, EDA rates and model coefficients answer different questions. Observed group differences are **unadjusted**, while Logistic Regression coefficients represent **conditional associations** after the other model features are included.

Therefore, a high observed attrition rate for a group should not automatically be interpreted as an independent effect of that group membership.

## Important Correlation Consideration

Several numerical variables are strongly correlated.

Examples from the relationship analysis include:

- MonthlyIncome vs JobLevel: approximately 0.95
- JobLevel vs TotalWorkingYears: approximately 0.78
- MonthlyIncome vs TotalWorkingYears: approximately 0.77
- YearsWithCurrManager vs YearsAtCompany: approximately 0.77
- YearsAtCompany vs YearsInCurrentRole: approximately 0.76

Because correlated variables can share predictive information, individual coefficient magnitudes should not be interpreted as a definitive ranking of causal importance.

This is especially important for variables such as income, job level, working years, and company tenure.

In a correlated feature set, the model may distribute predictive signal across related variables. Consequently, a coefficient's magnitude reflects the fitted model's conditional representation of the available information, not an isolated measure of real-world importance.

## Business Interpretation

The model provides an interpretable risk signal rather than an explanation of why an individual employee will leave.

The strongest positive associations suggest areas that could be investigated by HR, including:

- overtime exposure,
- frequently travelling roles,
- specific job-role groups,
- career progression variables,
- number of previous companies.

These should be treated as areas for further investigation rather than automatic intervention triggers.

For example, an employee predicted to be at higher attrition risk should not be labelled as likely to leave solely because the employee belongs to a particular job role.

A more appropriate use would be to treat the model as a **risk-screening signal** that can identify patterns for further review, subject to human judgment and organizational context.

## Responsible Use

The model should not be used as the sole basis for employment decisions.

Predicted attrition risk is a statistical estimate based on historical patterns in a fictional benchmark dataset. It does not establish employee intent, performance, loyalty, or cause of departure.

Any real-world application would require:

- privacy safeguards,
- appropriate access controls,
- fairness and subgroup evaluation,
- monitoring for data drift,
- human review,
- validation on organization-specific data,
- and explicit governance around employment-related decisions.

The model should also not be interpreted as evidence that an employee's personal characteristics cause attrition.

## Test-Set Separation

The held-out test set was reserved for the final performance evaluation.

Model interpretation was performed using the training partition, so held-out test outcomes were not used to calculate the reported coefficients, feature rankings, or explanatory statistics.

The final test results remain frozen and are documented separately in `docs/final_evaluation.md`.

This separation preserves the distinction between:

1. **Training data** — used to fit and interpret the final model.
2. **Cross-validation data** — used during model comparison and selection.
3. **Held-out test data** — used only for final performance evaluation and uncertainty estimation after the model and threshold were frozen.

## Generated Artifacts

- `reports/model_interpretation.csv`
- `reports/figures/model_coefficients.png`

## Limitations

1. Logistic Regression coefficients describe conditional statistical associations, not causal relationships.
2. One-hot encoded coefficients depend on the selected reference categories.
3. Numeric features were standardized, so numeric odds ratios represent changes per standardized unit rather than necessarily per original unit.
4. Correlated predictors can make individual coefficient interpretation unstable or difficult to isolate.
5. The dataset is a fictional IBM HR Analytics benchmark and may not represent a real organization's workforce.
6. The model coefficients describe the fitted model and should not be interpreted as causal effect sizes.
7. No test-set results are used to modify the frozen model.