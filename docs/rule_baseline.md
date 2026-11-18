# Simple Rules Baseline

## Purpose

The project includes a deliberately simple, interpretable baseline:

**flag = OverTime == "Yes" AND YearsAtCompany <= 3**

The rule is an **illustrative, post-hoc baseline**. It requires no model fitting and is evaluated on the same held-out 294-row test partition as the frozen Logistic Regression.

## Why it matters

A machine-learning model should be compared with an obvious business-rule heuristic. The comparison makes the coverage-versus-workload trade-off explicit.

| Method | Precision | Recall | F1 | Flag rate |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.3488 | **0.6383** | **0.4511** | 29.25% |
| Overtime + short-tenure rule | **0.5484** | 0.3617 | 0.4359 | **10.54%** |

The F1 values are comparable on only 47 positive test cases, so this should not be presented as a decisive ML win. The rule has higher precision and lower review workload; Logistic Regression has higher recall.

The rule cutoffs were motivated by exploratory analysis of the full dataset, including held-out rows. Therefore this is not a pre-registered or independently selected baseline. AP/ROC-AUC for the binary rule score are also not a fair ranking comparison with the continuous model scores.

## Reproduction

```bash
python src/rule_baseline.py
```

Output: `reports/rule_baseline_comparison.csv`
