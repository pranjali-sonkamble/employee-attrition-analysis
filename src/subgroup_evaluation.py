"""Evaluation-only subgroup diagnostics with bootstrap confidence intervals."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import get_split
from model_zoo import get_models
from preprocessing import build_model_pipeline

THRESHOLD = 0.50
N_BOOT = 2000
SEED = 42
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def subgroup_metrics(y_true, probabilities, threshold=THRESHOLD):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    negatives = tn + fp
    return {
        "rows": len(y_true),
        "attrition_cases": int(y_true.sum()),
        "attrition_rate": float(np.mean(y_true)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(fp / negatives if negatives else np.nan),
        "flag_rate": float(predictions.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true, probabilities, metric, n_boot=N_BOOT, seed=SEED):
    """Return a percentile 95% bootstrap interval for one subgroup metric."""
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    rng = np.random.default_rng(seed)
    values = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]

        if metric == "attrition_rate":
            values.append(float(yb.mean()))
        elif metric == "recall":
            positives = int(yb.sum())
            if positives == 0:
                continue
            pred = (pb >= THRESHOLD).astype(int)
            tp = int(((yb == 1) & (pred == 1)).sum())
            values.append(tp / positives)
        else:
            raise ValueError(metric)

    if not values:
        return np.nan, np.nan
    return tuple(np.quantile(values, [0.025, 0.975]))


def evaluate_group(group_name, group_values, y_test, probabilities):
    rows = []
    for value in sorted(group_values.dropna().unique()):
        mask = group_values == value
        y_group = y_test.loc[mask]
        p_group = probabilities[mask]
        metrics = subgroup_metrics(y_group, p_group)

        rate_low, rate_high = bootstrap_ci(
            y_group, p_group, "attrition_rate"
        )
        recall_low, recall_high = bootstrap_ci(
            y_group, p_group, "recall"
        )

        metrics.update({
            "group": group_name,
            "subgroup": str(value),
            "attrition_rate_ci95_low": rate_low,
            "attrition_rate_ci95_high": rate_high,
            "recall_ci95_low": recall_low,
            "recall_ci95_high": recall_high,
        })
        rows.append(metrics)
    return rows


def main():
    print("Subgroup evaluation")
    print("===================")
    print(f"Threshold: {THRESHOLD}")
    print("Model: frozen Logistic Regression")
    print(f"Bootstrap resamples per subgroup: {N_BOOT}")

    X_train, X_test, y_train, y_test = get_split()
    classifier, scale_numeric = get_models()["Logistic Regression"]

    pipeline = build_model_pipeline(
        classifier=classifier,
        scale_numeric=scale_numeric,
        use_smote=False,
    )
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results = []

    overall = subgroup_metrics(y_test, probabilities)
    rate_low, rate_high = bootstrap_ci(y_test, probabilities, "attrition_rate")
    recall_low, recall_high = bootstrap_ci(y_test, probabilities, "recall")
    overall.update({
        "group": "Overall",
        "subgroup": "All test employees",
        "attrition_rate_ci95_low": rate_low,
        "attrition_rate_ci95_high": rate_high,
        "recall_ci95_low": recall_low,
        "recall_ci95_high": recall_high,
    })
    results.append(overall)

    results.extend(evaluate_group("Gender", X_test["Gender"], y_test, probabilities))

    age_groups = pd.cut(
        X_test["Age"],
        bins=[0, 30, 36, 43, np.inf],
        labels=["18-30", "31-36", "37-43", "44+"],
        right=True,
    )
    results.extend(evaluate_group("Age group", age_groups, y_test, probabilities))

    columns = [
        "group", "subgroup", "rows", "attrition_cases", "attrition_rate",
        "attrition_rate_ci95_low", "attrition_rate_ci95_high", "precision",
        "recall", "recall_ci95_low", "recall_ci95_high",
        "false_positive_rate", "flag_rate", "tn", "fp", "fn", "tp",
    ]
    results_df = pd.DataFrame(results)[columns]
    output_path = OUTPUT_DIR / "subgroup_evaluation.csv"
    results_df.to_csv(output_path, index=False)

    print(results_df.to_string(index=False))
    age_44 = results_df[
        (results_df["group"] == "Age group")
        & (results_df["subgroup"] == "44+")
    ]
    if not age_44.empty:
        row = age_44.iloc[0]
        print(
            f"Age 44+ recall: {int(row.tp)} of {int(row.attrition_cases)} "
            f"attrition cases ({row.recall:.2%}); bootstrap 95% CI "
            f"{row.recall_ci95_low:.2%}–{row.recall_ci95_high:.2%}."
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
