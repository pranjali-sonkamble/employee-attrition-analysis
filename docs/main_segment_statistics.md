# Main Segment Statistics

The project reports uncertainty and association tests for three main descriptive segments:

- Overtime
- Sales Representative role
- Early tenure (YearsAtCompany <= 3)

For each segment, the script reports segment size, observed attrition cases, observed attrition rate, Wilson 95% confidence interval for the rate, chi-square statistic, degrees of freedom, chi-square p-value, and minimum expected cell count.

The chi-square test compares segment membership with observed attrition status. It is an association test, not a causal test.

## Reproduction

    python src/segment_statistics.py

Output: reports/main_segment_statistics.csv

The confidence intervals describe uncertainty around the observed benchmark rates. The p-values should not be interpreted as evidence that the segment causes attrition, especially because the dataset is a fictional benchmark and multiple descriptive comparisons are being examined.
