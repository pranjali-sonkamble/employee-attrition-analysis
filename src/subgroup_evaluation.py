"""
Subgroup evaluation of the frozen Logistic Regression model.

This is an evaluation-only analysis.
It does not tune or modify the frozen model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

from data_prep import get_split
from preprocessing import build_model_pipeline


RANDOM_STATE = 42
THRESHOLD = 0.50

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def subgroup_metrics(y_true, probabilities, threshold=0.50):
    """Calculate threshold-based metrics for one subgroup."""

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    actual_positive = tp + fn
    actual_negative = tn + fp
    predicted_positive = tp + fp

    return {
        "rows": len(y_true),
        "attrition_cases": int(actual_positive),
        "attrition_rate": float(np.mean(y_true)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "false_positive_rate": float(
            fp / actual_negative if actual_negative else np.nan
        ),
        "flag_rate": float(
            predicted_positive / len(y_true)
            if len(y_true)
            else np.nan
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_group(
    group_name,
    group_values,
    y_test,
    probabilities,
):
    """Evaluate every category within one subgroup variable."""

    rows = []

    for value in sorted(group_values.dropna().unique()):
        mask = group_values == value

        metrics = subgroup_metrics(
            y_test.loc[mask],
            probabilities[mask],
            threshold=THRESHOLD,
        )

        metrics["group"] = group_name
        metrics["subgroup"] = str(value)

        rows.append(metrics)

    return rows


def main():
    print("Subgroup evaluation")
    print("===================")
    print(f"Threshold: {THRESHOLD}")
    print("Model: frozen Logistic Regression")
    print("Purpose: evaluation only; no tuning or model modification")

    X_train, X_test, y_train, y_test = get_split()

    pipeline = classifier = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)

    pipeline = build_model_pipeline(
    classifier=classifier,
)

    # Refit the already-frozen model configuration on training data only.
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results = []

    # ---------------------------------------------------------
    # Overall test-set performance
    # ---------------------------------------------------------
    overall = subgroup_metrics(
        y_test,
        probabilities,
        threshold=THRESHOLD,
    )

    overall["group"] = "Overall"
    overall["subgroup"] = "All test employees"

    results.append(overall)

    # ---------------------------------------------------------
    # Gender
    # ---------------------------------------------------------
    results.extend(
        evaluate_group(
            "Gender",
            X_test["Gender"],
            y_test,
            probabilities,
        )
    )

    # ---------------------------------------------------------
    # Age groups
    #
    # These bands match the earlier EDA grouping:
    # 18-30, 31-36, 37-43, 44+
    # ---------------------------------------------------------
    age_groups = pd.cut(
        X_test["Age"],
        bins=[0, 30, 36, 43, np.inf],
        labels=["18-30", "31-36", "37-43", "44+"],
        right=True,
    )

    results.extend(
        evaluate_group(
            "Age group",
            age_groups,
            y_test,
            probabilities,
        )
    )

    results_df = pd.DataFrame(results)

    column_order = [
        "group",
        "subgroup",
        "rows",
        "attrition_cases",
        "attrition_rate",
        "precision",
        "recall",
        "false_positive_rate",
        "flag_rate",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    results_df = results_df[column_order]

    output_path = OUTPUT_DIR / "subgroup_evaluation.csv"

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(results_df.to_string(index=False))
    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()