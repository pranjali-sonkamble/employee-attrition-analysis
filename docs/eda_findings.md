\# Exploratory Data Analysis — Findings



\## 1. Dataset overview



\- \*\*Records:\*\* 1,470 employees

\- \*\*Original columns:\*\* 35

\- \*\*Attrition = Yes:\*\* 237 employees (16.12%)

\- \*\*Attrition = No:\*\* 1,233 employees (83.88%)

\- \*\*Missing values:\*\* None detected during initial inspection

\- \*\*Duplicate rows:\*\* None detected during initial inspection



\## 2. Key descriptive findings



\### Work patterns

\- Employees with overtime had an attrition rate of \*\*30.53%\*\*, compared with \*\*10.44%\*\* among employees without overtime.

\- Employees who travelled frequently had an attrition rate of \*\*24.91%\*\*.



\### Employee characteristics

\- Employees in the Single marital-status group had an attrition rate of \*\*25.53%\*\*.

\- Employees aged 30 or below had an attrition rate of \*\*25.91%\*\* in the age-band analysis.



\### Job roles

\- Sales Representatives had the highest observed role-level attrition rate at \*\*39.76%\*\*.

\- Research Directors had the lowest observed role-level rate at \*\*2.50%\*\*.



\### Income and tenure

\- The lowest monthly-income band had an attrition rate of \*\*29.27%\*\*.

\- The shortest-tenure band (up to 3 years in the company) had an attrition rate of \*\*25.96%\*\*.

\- The longest total-working-years band (over 15 years in the reported grouping) had an attrition rate of \*\*8.71%\*\*.



\## 3. Data and modelling implications



1\. Attrition is an imbalanced target: only 16.12% of records have `Attrition = Yes`.

2\. Accuracy alone is insufficient for evaluating a predictive model. Recall, precision, F1, PR-AUC, and the confusion matrix will also be considered.

3\. `EmployeeNumber` is a unique identifier and should not be used as a predictive feature.

4\. `EmployeeCount`, `Over18`, and `StandardHours` are constant in this dataset and should be excluded from modelling.

5\. Any learned preprocessing, feature selection, resampling, or model tuning must be fitted using training data only. The final test set must remain separate from model selection.



\## 4. Limitations



\- These findings describe associations in the supplied dataset; they do not establish causation.

\- Group sizes differ, so percentages should be interpreted alongside record counts.

\- This dataset does not establish that the same patterns apply to a real organisation.

\- Predictions from this project are for learning and analytical demonstration, not for making employment decisions about individuals.



\## 5. Reproducibility



The EDA script is `src/explore\_data.py`. It generates the summary in `reports/eda\_summary.txt`, the categorical and numeric rate tables in `reports/`, and figures in `reports/figures/`.



