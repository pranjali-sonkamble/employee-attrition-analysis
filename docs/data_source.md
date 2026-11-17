# Dataset Source and Provenance

## Dataset details

- **Dataset:** IBM HR Analytics Employee Attrition & Performance
- **Platform:** Kaggle
- **Uploader:** pavansubhasht
- **Source:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- **Access date recorded in the original project notes:** 21 October 2026
- **Dataset description:** The Kaggle page describes this as a fictional dataset created by IBM data scientists.

The repository does not contain the raw CSV. Follow the download instructions in [data/README.md](../data/README.md).

## Licence and redistribution

The Kaggle page was recorded as showing Open Database and Database Contents licensing:

- [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/)
- [Database Contents License (DbCL) 1.0](https://opendatacommons.org/licenses/dbcl/1-0/)

**Redistribution status: not independently confirmed.** The raw CSV remains excluded from this public repository. Review the licence texts and source-page terms before redistributing the dataset.

## Initial data inspection

The committed local inspection output records:

- **Rows:** 1,470
- **Columns:** 35
- **Missing values:** 0
- **Duplicate rows:** 0
- **Constant columns:** `EmployeeCount`, `Over18`, and `StandardHours`
- **Unique identifier:** `EmployeeNumber` (1,470 unique values)

### Target distribution

| Attrition | Count | Proportion |
|---|---:|---:|
| No | 1,233 | 83.88% |
| Yes | 237 | 16.12% |

These values describe the local CSV used for this project; they are not claims about a real employer's workforce.

## Validation and analysis status

The dataset inspection, column validation, exploratory analysis, model comparison, threshold evaluation, final held-out test evaluation, uncertainty estimation, model interpretation, and subgroup diagnostics are documented in the repository.

Key evidence:

- [reports/data_inspection.txt](../reports/data_inspection.txt)
- [reports/column_validation.txt](../reports/column_validation.txt)
- [reports/eda_summary.txt](../reports/eda_summary.txt)
- [docs/eda_findings.md](eda_findings.md)
- [docs/final_evaluation.md](final_evaluation.md)
- [docs/subgroup_evaluation.md](subgroup_evaluation.md)

The project remains an educational analytical prototype. Because the data is fictional and the analysis is observational, the findings must not be presented as verified facts about a real organization or used as the sole basis for employment decisions.
