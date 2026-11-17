# Sensitive-Attribute Ablation

## Purpose

Compare the frozen Logistic Regression specification with an otherwise identical model that excludes Age, Gender, and MaritalStatus from model inputs.

The same stratified train/test split, preprocessing approach, classifier configuration, random seed, and 0.50 threshold are used for both models. The three attributes remain available only for subgroup grouping in the ablated evaluation.

## Results

Run python src/sensitive_attribute_ablation.py to regenerate the tables below. The committed CSV files are the source of the reported values.

### Overall test-set metrics

| Model | Average Precision | ROC-AUC | Precision | Recall | Flag rate |
|---|---:|---:|---:|---:|---:|
| Full model | 0.5612 | 0.8032 | 0.3488 | 0.6383 | 29.25% |
| Without Age/Gender/MaritalStatus | 0.5433 | 0.7801 | 0.3659 | 0.6383 | 27.89% |

### Subgroup results

The subgroup table compares the same fixed threshold within Gender, MaritalStatus, and the project's predefined age bands. These are descriptive diagnostics, not a fairness certification.

See reports/sensitive_attribute_ablation_subgroups.csv for the complete results.

## Interpretation

If the ablated model changes predictive performance or subgroup metrics materially, that difference should be treated as evidence that these attributes contribute predictive information in this benchmark. It does not establish causality, fairness, or what would happen in another organization.

Because the dataset is a fictional benchmark and the test partition contains only 47 attrition cases, small subgroup differences should not be over-interpreted.

## Responsible-use note

The ablation is an analytical diagnostic. It does not establish that excluding these attributes would make the model appropriate for employment decisions. Any real HR application would require organization-specific validation, privacy safeguards, a predefined fairness framework, governance, and human oversight.
