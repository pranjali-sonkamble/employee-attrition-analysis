"""Evaluate a simple deterministic overtime + short-tenure baseline.

The rule is an illustrative, post-hoc baseline:
flag attrition risk when OverTime == "Yes" AND YearsAtCompany <= 3.
It is evaluated on the same held-out test partition as the frozen Logistic
Regression model so the comparison is apples-to-apples.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import get_split
from model_zoo import get_models
from preprocessing import build_model_pipeline

THRESHOLD = 0.50
SHORT_TENURE_YEARS = 3


def rule_predictions(X):
    """Return 1 when both overtime and short tenure conditions are met."""
    return (
        X["OverTime"].eq("Yes")
        & X["YearsAtCompany"].le(SHORT_TENURE_YEARS)
    ).astype(int).to_numpy()


def evaluate_binary_scores(y_true, scores):
    """Evaluate binary or probability scores with a 0.50 decision threshold."""
    predictions = (scores >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "average_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "flag_rate": float(predictions.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    X_train, X_test, y_train, y_test = get_split()

    classifier, scale_numeric = get_models()["Logistic Regression"]
    pipeline = build_model_pipeline(
        clone(classifier),
        scale_numeric=scale_numeric,
        use_smote=False,
    )
    pipeline.fit(X_train, y_train)

    ml_scores = pipeline.predict_proba(X_test)[:, 1]
    rule_scores = rule_predictions(X_test)

    rows = []
    for model_name, scores in [
        ("Logistic Regression", ml_scores),
        ("Overtime + short tenure rule", rule_scores),
    ]:
        row = evaluate_binary_scores(y_test.to_numpy(), scores)
        row["method"] = model_name
        rows.append(row)

    result = pd.DataFrame(rows)[[
        "method",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "flag_rate",
        "tn",
        "fp",
        "fn",
        "tp",
    ]]

    Path("reports").mkdir(exist_ok=True)
    output = Path("reports/rule_baseline_comparison.csv")
    result.to_csv(output, index=False)

    rule_flagged = int(rule_scores.sum())
    print("Simple rules baseline")
    print("=====================")
    print(
        "Rule: OverTime == 'Yes' AND YearsAtCompany <= "
        f"{SHORT_TENURE_YEARS}"
    )
    print(f"Rule flags: {rule_flagged}/{len(rule_scores)}")
    print()
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
