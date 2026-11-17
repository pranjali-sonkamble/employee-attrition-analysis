# Employee Attrition Analysis & Prediction

An end-to-end analytical prototype for understanding employee attrition patterns and evaluating a machine-learning model for attrition risk prediction using the IBM HR Analytics Employee Attrition & Performance benchmark dataset.

> **Project status:** Portfolio-ready analytical prototype
> **Primary focus:** Data analysis, model evaluation, interpretation, uncertainty, subgroup diagnostics, and reproducibility
> **Important:** This project is not a production HR decision system and should not be used as the sole basis for employment decisions.

---

## 1. Project Overview

Employee attrition is an important workforce analytics problem because organizations need to understand where turnover risk is concentrated and which employee/workplace characteristics are associated with attrition.

This project follows a complete analytical workflow:

**Raw Dataset â†’ Data Validation â†’ EDA â†’ Preprocessing â†’ Model Comparison â†’ Threshold Evaluation â†’ Final Held-Out Test â†’ Uncertainty Analysis â†’ Model Interpretation â†’ Subgroup Evaluation â†’ Regression Tests**

The project deliberately separates model development from final evaluation so that the held-out test set is not used to tune the model or select the final threshold.

---

## 2. Objectives

The project has four main objectives:

1. **Understand the dataset**
   - Validate schema, identifiers, data types, constant columns, and logical consistency.
   - Explore employee demographics, job characteristics, compensation, satisfaction, tenure, and attrition.

2. **Build and compare predictive models**
   - Use reproducible preprocessing and stratified evaluation.
   - Compare candidate models using repeated cross-validation.
   - Select a final model before evaluating the held-out test set.

3. **Evaluate prediction quality**
   - Measure ranking performance and classification performance.
   - Evaluate the selected threshold using out-of-fold predictions.
   - Perform final evaluation once on the untouched test set.

4. **Assess limitations and reliability**
   - Quantify test-set uncertainty with bootstrap confidence intervals.
   - Interpret model coefficients carefully.
   - Check performance across descriptive subgroups.
   - Add regression tests to protect the data/model pipeline.

---

## 3. Dataset

**Dataset:** IBM HR Analytics Employee Attrition & Performance

The dataset is a fictional benchmark dataset containing employee-level HR attributes and an `Attrition` target.

Local inspection:

- **Rows:** 1,470
- **Columns:** 35
- **Target:** `Attrition`
- **Target classes:** `Yes` / `No`

The dataset source and licensing information are documented in:

- `docs/data_source.md`
- `reports/data_inspection.txt`

The raw dataset is intentionally excluded from Git through `.gitignore`.

---

## 4. Data Validation

Before modeling, the dataset was inspected for structural and logical issues.

Validation included:

- Dataset shape and schema inspection
- Unique employee identifier validation
- Constant-column detection
- Coded ordinal variable inspection
- Numeric summary statistics
- Logical consistency checks
- Target validation
- Train/test separation checks

### Identifier validation

`EmployeeNumber` is unique across all 1,470 records and is treated as an identifier rather than a predictive feature.

### Constant columns

The following columns contain no useful variation and are excluded:

- `EmployeeCount`
- `Over18`
- `StandardHours`

`EmployeeNumber` is also excluded as an identifier.

The resulting modeling dataset contains **30 candidate features** before preprocessing.

---

## 5. Exploratory Data Analysis

The EDA examines relationships between attrition and employee/job characteristics including:

- Age
- Business travel
- Department
- Education
- Education field
- Job level
- Job role
- Monthly income
- Overtime
- Marital status
- Job satisfaction
- Environment satisfaction
- Years at company
- Years in current role
- Years since last promotion
- Years with current manager
- Number of companies worked

EDA findings and supporting artifacts are documented in:

`docs/eda_findings.md`

The analysis is descriptive and does not interpret observational relationships as causal effects.

---

## 6. Preprocessing

The preprocessing pipeline is implemented with reusable scikit-learn / imbalanced-learn components.

### Feature groups

**Categorical features**

- BusinessTravel
- Department
- EducationField
- Gender
- JobRole
- MaritalStatus
- OverTime

**Ordinal features**

- Education
- EnvironmentSatisfaction
- JobInvolvement
- JobSatisfaction
- RelationshipSatisfaction
- WorkLifeBalance
- StockOptionLevel
- PerformanceRating

**Numeric features**

- Age
- DailyRate
- DistanceFromHome
- HourlyRate
- JobLevel
- MonthlyIncome
- MonthlyRate
- NumCompaniesWorked
- PercentSalaryHike
- TotalWorkingYears
- TrainingTimesLastYear
- YearsAtCompany
- YearsInCurrentRole
- YearsSinceLastPromotion
- YearsWithCurrManager

### Pipeline behavior

- Numeric missing values â†’ median imputation
- Categorical missing values â†’ most-frequent imputation
- Categorical variables â†’ one-hot encoding
- Numeric variables â†’ scaling where required
- Unknown categorical levels â†’ safely ignored during transformation
- Optional SMOTE support is available in the modeling pipeline

The preprocessing logic is implemented in:

`src/preprocessing.py`

---

## 7. Train/Test Design

The dataset is split using:

- **Test size:** 20%
- **Stratification:** Yes
- **Random state:** 42

This produces:

- **Training rows:** 1,176
- **Held-out test rows:** 294

The test set is reserved for final evaluation.

No model tuning or threshold selection is performed using the final test labels.

---

## 8. Model Comparison

Candidate models are evaluated using **repeated stratified 5-fold cross-validation with 3 repeats**.

The model comparison stage is performed before the final held-out test evaluation.

The selected final model is:

### Logistic Regression

Repeated-CV performance of the selected model:

| Metric | Mean CV Score |
|---|---:|
| Average Precision | **0.6130** |
| ROC-AUC | **0.8251** |

The model comparison artifacts are stored in the `reports/` directory.

---

## 9. Threshold Evaluation

Because attrition prediction is an imbalanced binary classification problem, threshold choice affects the precision/recall trade-off.

The threshold evaluation uses **out-of-fold predictions** rather than the final test set.

The final evaluation threshold was fixed at:

**0.50**

The threshold is therefore treated as a predetermined evaluation setting for the final test rather than something optimized on the test data.

Detailed threshold analysis:

`docs/threshold_evaluation.md`

---

## 10. Final Held-Out Test Evaluation

The final Logistic Regression model was evaluated once on the untouched 294-row test set at the fixed threshold of 0.50.

### Test-set results

| Metric | Result |
|---|---:|
| Average Precision | **0.5612** |
| ROC-AUC | **0.8032** |
| Precision | **0.3488** |
| Recall | **0.6383** |
| F1-score | **0.4511** |

### Confusion matrix

|  | Predicted No | Predicted Yes |
|---|---:|---:|
| **Actual No** | 191 | 56 |
| **Actual Yes** | 17 | 30 |

Therefore:

- **True Negatives:** 191
- **False Positives:** 56
- **False Negatives:** 17
- **True Positives:** 30

The model identifies approximately 64% of the attrition cases in this held-out test sample, while generating a substantial number of false positives.

This trade-off is important: the model is better interpreted as a **risk-ranking / screening prototype** than as a definitive predictor of which employees will leave.

Detailed results:

`docs/final_evaluation.md`

---

## 11. Test-Set Uncertainty

Point estimates alone can hide uncertainty, especially with a relatively small number of attrition cases.

A bootstrap analysis was therefore performed on the frozen 294-row test set.

Configuration:

- **Bootstrap resamples:** 2,000 valid resamples
- **Random seed:** 42
- **Model:** frozen
- **Threshold:** frozen
- **Test rows:** 294

### 95% bootstrap intervals

| Metric | Estimate | 95% CI |
|---|---:|---:|
| Average Precision | 0.561 | 0.425â€“0.689 |
| ROC-AUC | 0.803 | 0.727â€“0.873 |
| Precision | 0.349 | 0.247â€“0.453 |
| Recall | 0.638 | 0.489â€“0.778 |
| Recall @ Top 10% | 0.362 | 0.260â€“0.475 |
| Recall @ Top 20% | 0.532 | 0.420â€“0.676 |
| Recall @ Top 30% | 0.638 | 0.524â€“0.810 |

The bootstrap analysis does **not** modify the model or threshold. It quantifies uncertainty around the already-frozen test evaluation.

Detailed results:

`reports/test_uncertainty.csv`

---

## 12. Model Interpretation

The final Logistic Regression model uses **51 transformed features** after preprocessing.

Because the model is linear in the transformed feature space, coefficients can be examined to understand which features are associated with higher or lower predicted log-odds, conditional on the other features in the model.

### Strongest positive coefficients

| Feature | Coefficient | Odds Ratio |
|---|---:|---:|
| JobRole â€” Laboratory Technician | 1.1943 | 3.301 |
| JobRole â€” Sales Representative | 1.1206 | 3.067 |
| BusinessTravel â€” Travel Frequently | 0.7876 | 2.198 |
| OverTime â€” Yes | 0.7615 | 2.142 |
| EducationField â€” Human Resources | 0.7211 | 2.057 |

### Strongest negative coefficients

| Feature | Coefficient | Odds Ratio |
|---|---:|---:|
| JobRole â€” Research Director | -1.4657 | 0.231 |
| BusinessTravel â€” Non-Travel | -0.9939 | 0.370 |
| OverTime â€” No | -0.8928 | 0.410 |
| EducationField â€” Other | -0.7981 | 0.450 |
| JobRole â€” Healthcare Representative | -0.6472 | 0.524 |

### Interpretation caution

These coefficients represent **conditional statistical associations**, not causal effects.

For categorical variables, coefficients are interpreted relative to the corresponding reference category created by one-hot encoding.

Correlated predictors can also affect coefficient magnitude and stability. Therefore, the coefficients should not be interpreted as evidence that changing a particular employee characteristic would cause attrition to increase or decrease.

Detailed interpretation:

`docs/model_interpretation.md`

---

## 13. Subgroup Evaluation

The frozen model was evaluated descriptively across selected gender and age subgroups.

### Gender

| Group | Rows | Attrition Cases | Precision | Recall | FPR | Flag Rate |
|---|---:|---:|---:|---:|---:|---:|
| Female | 116 | 16 | 0.324 | 0.688 | 0.230 | 0.293 |
| Male | 178 | 31 | 0.365 | 0.613 | 0.224 | 0.292 |

### Age groups

| Age Group | Rows | Attrition Cases | Precision | Recall | FPR | Flag Rate |
|---|---:|---:|---:|---:|---:|---:|
| 18â€“30 | 77 | 18 | 0.438 | 0.778 | 0.305 | 0.416 |
| 31â€“36 | 84 | 11 | 0.258 | 0.727 | 0.315 | 0.369 |
| 37â€“43 | 72 | 7 | 0.357 | 0.714 | 0.138 | 0.194 |
| 44+ | 61 | 11 | 0.333 | 0.273 | 0.120 | 0.148 |

The subgroup results show that model performance is **not uniform across all groups**, particularly across age bands.

However, these descriptive results do **not by themselves establish fairness, discrimination, or causal bias**. Some groups contain relatively few attrition cases, so estimates can be unstable.

Subgroup methodology and limitations:

`docs/subgroup_evaluation.md`

---

## 14. Responsible Use

This project is an analytical prototype using a fictional benchmark dataset.

It should **not** be used as the sole basis for:

- Hiring decisions
- Termination decisions
- Promotion decisions
- Compensation decisions
- Employee disciplinary action
- Individual HR interventions

A real deployment would require additional work, including:

- Organization-specific validation
- Privacy and access controls
- Fairness and impact assessment
- Monitoring for data drift and performance degradation
- Calibration analysis
- Uncertainty-aware decision policies
- Human oversight
- Appropriate HR/legal governance
- Documentation of permissible use cases

The model should be treated as a decision-support or analytical tool rather than an automated decision-maker.

---

## 15. Regression Tests

The project includes automated regression tests covering the data and modeling pipeline.

Current test suite:

**10 tests passed**

```text
pytest -q
..........                                                               [100%]
10 passed in 2.42s
```

The tests cover:

- Expected dataset shape
- Target encoding
- Dropped identifier/constant columns
- Expected feature count
- Reproducible train/test splitting
- Split size and stratification
- Preprocessor structure
- Missing-value handling after transformation
- Model pipeline structure
- Model fitting and probability generation
- Binary prediction behavior

Tests are located in:

`tests/test_data_pipeline.py`

---

## 16. Reproducibility

The project uses deterministic settings where appropriate, including:

- Random state: **42**
- Stratified train/test split
- Repeated stratified cross-validation
- Fixed evaluation threshold
- Bootstrap seed: **42**

### Environment

The project was developed with:

- Python 3.10
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- matplotlib
- pytest

Dependencies are recorded in the project requirements files.

### Run tests

```bash
pytest -q
```

### Run key analysis stages

```bash
python src/data_prep.py
python src/compare_models.py
python src/evaluate_thresholds.py
python src/final_evaluation.py
python src/test_uncertainty.py
python src/model_interpretation.py
python src/subgroup_evaluation.py
```

Run the scripts from the project root with the virtual environment activated.

---

## 17. Project Structure

```text
employee-attrition-analysis/
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â””â”€â”€ processed/
â”‚
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ project_charter.md
â”‚   â”œâ”€â”€ data_source.md
â”‚   â”œâ”€â”€ eda_findings.md
â”‚   â”œâ”€â”€ threshold_evaluation.md
â”‚   â”œâ”€â”€ final_evaluation.md
â”‚   â”œâ”€â”€ model_interpretation.md
â”‚   â””â”€â”€ subgroup_evaluation.md
â”‚
â”œâ”€â”€ models/
â”‚
â”œâ”€â”€ notebooks/
â”‚
â”œâ”€â”€ reports/
â”‚   â”œâ”€â”€ figures/
â”‚   â”œâ”€â”€ data_inspection.txt
â”‚   â”œâ”€â”€ column_validation.txt
â”‚   â”œâ”€â”€ model_interpretation.csv
â”‚   â”œâ”€â”€ subgroup_evaluation.csv
â”‚   â””â”€â”€ test_uncertainty.csv
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data_prep.py
â”‚   â”œâ”€â”€ preprocessing.py
â”‚   â”œâ”€â”€ model_zoo.py
â”‚   â”œâ”€â”€ compare_models.py
â”‚   â”œâ”€â”€ evaluate_thresholds.py
â”‚   â”œâ”€â”€ final_evaluation.py
â”‚   â”œâ”€â”€ test_uncertainty.py
â”‚   â”œâ”€â”€ model_interpretation.py
â”‚   â””â”€â”€ subgroup_evaluation.py
â”‚
â”œâ”€â”€ tests/
â”‚   â””â”€â”€ test_data_pipeline.py
â”‚
â”œâ”€â”€ .gitignore
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ requirements-lock.txt
â””â”€â”€ README.md
```

---

## 18. Key Takeaways

### What the project demonstrates

- End-to-end employee attrition analytics
- Reproducible data preparation
- Structured exploratory analysis
- Leakage-aware train/test separation
- Reusable preprocessing pipelines
- Repeated cross-validation
- Out-of-fold threshold evaluation
- Final held-out test evaluation
- Bootstrap uncertainty estimation
- Interpretable Logistic Regression
- Subgroup performance diagnostics
- Automated regression testing
- Responsible-use documentation

### Final model snapshot

| Item | Result |
|---|---|
| Final model | Logistic Regression |
| Test rows | 294 |
| Threshold | 0.50 |
| ROC-AUC | 0.8032 |
| Average Precision | 0.5612 |
| Precision | 0.3488 |
| Recall | 0.6383 |
| F1 | 0.4511 |
| TP / FP / TN / FN | 30 / 56 / 191 / 17 |
| Bootstrap resamples | 2,000 |
| Regression tests | 10 passed |

The most important conclusion is not simply that the model achieves a particular score. The project demonstrates a disciplined evaluation process: model selection is separated from final testing, uncertainty is quantified, interpretation is qualified, subgroup behavior is examined, and the pipeline is protected with automated tests.

---

## 19. Limitations

The project has several important limitations:

1. The dataset is fictional and may not represent real organizational workforce behavior.
2. The sample size is modest for subgroup-level analysis.
3. Attrition is an observational outcome, so model associations should not be interpreted causally.
4. The final model is evaluated on a single held-out test split.
5. Performance may change substantially on a different organization, time period, or population.
6. The chosen threshold reflects a specific evaluation trade-off and may not match a real organization's operational costs.
7. The model is not calibrated or validated for production HR use.

These limitations are part of the project's conclusions rather than issues hidden from the evaluation.

---

## 20. Conclusion

This project provides a reproducible framework for employee attrition analysis and predictive modeling while explicitly documenting model performance, uncertainty, interpretation, subgroup behavior, and limitations.

The final Logistic Regression model demonstrates useful ranking and screening performance on the benchmark test set, but its precision/recall trade-off and dataset limitations make it unsuitable for unsupervised real-world employment decisions.

The project therefore emphasizes **transparent evaluation and responsible analytical use** rather than presenting attrition prediction as a definitive HR decision system.
