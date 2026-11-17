# Exploratory Data Analysis - Findings

## 1. Dataset overview

- **Records:** 1,470 employees
- **Original columns:** 35
- **Attrition = Yes:** 237 employees (16.12%)
- **Attrition = No:** 1,233 employees (83.88%)
- **Missing values:** None detected during initial inspection
- **Duplicate rows:** None detected during initial inspection

## 2. Key descriptive findings

### Work patterns

- Employees with overtime had an attrition rate of **30.53%**, compared with **10.44%** among employees without overtime.
- Employees who travelled frequently had an attrition rate of **24.91%**.

### Employee characteristics

- Employees in the Single marital-status group had an attrition rate of **25.53%**.
- Employees aged 30 or below had an attrition rate of **25.91%**.
- Attrition rates across the age bands were **25.91%** for ages up to 30, **16.02%** for ages 31 to 36, **8.92%** for ages 37 to 43, and **12.10%** for ages 44 to 60. The pattern is not monotonic: the rate falls through the 37 to 43 band and then rises in the oldest band.

### Job roles

- Sales Representatives had the highest observed role-level attrition rate at **39.76%** (83 employees).
- Research Directors had the lowest observed role-level rate at **2.50%** (80 employees).

### Income and tenure

- The lowest monthly-income band had an attrition rate of **29.27%**.
- Employees with `YearsAtCompany` **≤ 3 years** had an attrition rate of **25.96%**. This is the project's definition of the early-tenure group.
- The longest total-working-years band, covering more than 15 years in the reported grouping, had an attrition rate of **8.71%**.

### Sample-size cautions

- The Human Resources department contains 63 records, and the Human Resources education-field group contains 27 records. Their observed attrition rates are based on relatively small groups.
- The Sales Representative attrition rate of 39.76% is based on 83 employees and should be interpreted alongside the group size.
- Some ordinal levels also have relatively small samples: JobInvolvement level 1 (83 records), WorkLifeBalance level 1 (80 records), StockOptionLevel level 3 (85 records), and Education level 5 (48 records).
- The 17.65% attrition rate for StockOptionLevel level 3 represents approximately 15 employees. It should not be interpreted as a reliable trend on its own.
- Group-level percentages are descriptive estimates and may be less stable for smaller groups.

## 3. Correlation and ordinal-feature findings

### Numeric-feature correlations

The strongest absolute Pearson correlations observed among the examined numeric features were:

| Feature pair | Correlation |
|---|---:|
| MonthlyIncome and JobLevel | 0.95 |
| JobLevel and TotalWorkingYears | 0.78 |
| MonthlyIncome and TotalWorkingYears | 0.77 |
| YearsWithCurrManager and YearsAtCompany | 0.77 |
| YearsAtCompany and YearsInCurrentRole | 0.76 |

These correlations indicate that some predictors contain overlapping information. In particular, the strong correlation between MonthlyIncome and JobLevel should be considered when interpreting Logistic Regression coefficients.

Income, job level, and total working years form a correlated group of features. The tenure-related variables also show strong correlations with one another. Correlated features may share or divide predictive importance, so feature importance results should be interpreted with these relationships in mind. Correlation alone does not establish causation or require automatic feature removal.

DailyRate, HourlyRate, and MonthlyRate show weak pairwise linear correlations with the other examined numeric features in this dataset. This observation alone does not establish that these variables are irrelevant to attrition. Their predictive usefulness will be evaluated during modelling, with permutation importance considered in Phase 8.

### Ordinal-feature patterns

- **EnvironmentSatisfaction:** Attrition was 25.35% at level 1 and 13.45% at level 4.
- **JobInvolvement:** Attrition was 33.73% at level 1 and 9.03% at level 4.
- **JobSatisfaction:** Attrition was 22.84% at level 1 and 11.33% at level 4.
- **WorkLifeBalance:** Attrition was 31.25% at level 1, 14.22% at level 3, and 17.65% at level 4.
- **StockOptionLevel:** Attrition was 24.41% at level 0 (631 records), 9.40% at level 1 (596 records), 7.59% at level 2 (158 records), and 17.65% at level 3 (85 records).
- **PerformanceRating:** Attrition was 16.08% at level 3 and 16.37% at level 4.

The StockOptionLevel pattern is non-monotonic: the observed rate decreases from level 0 through level 2 and rises at level 3. The level 3 estimate is based on only 85 records.

These patterns are descriptive associations. They do not demonstrate that changing a feature will cause attrition to change. Rates for levels with relatively few records require particular caution.

## 4. Data and modelling implications

1. Attrition is an imbalanced target: only 16.12% of records have `Attrition = Yes`.
2. Accuracy alone is insufficient for evaluating a predictive model. Recall, precision, F1, PR-AUC, and the confusion matrix will also be considered.
3. `EmployeeNumber` is a unique identifier and should not be used as a predictive feature.
4. `EmployeeCount`, `Over18`, and `StandardHours` are constant in this dataset and should be excluded from modelling.
5. Numeric predictors are correlated. Their effects, coefficients, and feature importance should be interpreted carefully during modelling.
6. Any learned preprocessing, feature selection, resampling, or model tuning must be fitted using training data only. The final test set must remain separate from model selection.

## 5. Fairness and responsible use

Gender and marital status are included in descriptive analysis to understand observed group-level patterns. They will also be examined during the planned fairness review in Phase 8. These attributes and their observed associations must not be used to justify adverse employment decisions about individuals.

The project is an educational analytical demonstration, not a validated employment decision system.

## 6. Limitations

- These findings describe associations in the supplied dataset; they do not establish causation.
- Group sizes differ, so percentages should be interpreted alongside record counts.
- Pearson correlations describe linear relationships between numeric features and do not establish causal relationships.
- This dataset does not establish that the same patterns apply to a real organisation.
- Predictions from this project are for learning and analytical demonstration, not for making employment decisions about individuals.

## 7. Reproducibility

- Core EDA script: `src/explore_data.py`
- Core EDA summary: `reports/eda_summary.txt`
- Relationship EDA script: `src/explore_relationships.py`
- Relationship summary: `reports/relationship_summary.txt`
- Figures: `reports/figures/`

The scripts read the local raw dataset and generate descriptive reports and figures. Learned preprocessing and predictive modelling are handled separately from this exploratory analysis.