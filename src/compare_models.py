from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Make imports work when this script is run as:
# python src\\compare_models.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import get_split
from model_zoo import get_models
from preprocessing import build_model_pipeline


N_SPLITS = 5
N_REPEATS = 3
BASE_SEED = 42
CAPACITY_FRACTIONS = [0.10, 0.20, 0.30]


def top_k_metrics(y, probabilities, fraction):
    """Return recall and precision among the highest-scored fraction."""
    n_flagged = int(round(len(y) * fraction))
    if n_flagged < 1:
        raise ValueError("Capacity fraction flags fewer than one row.")

    ranked_indices = np.argsort(-probabilities, kind="stable")[:n_flagged]
    true_positives = int(y[ranked_indices].sum())
    total_leavers = int(y.sum())

    recall = true_positives / total_leavers if total_leavers else np.nan
    precision = true_positives / n_flagged
    return recall, precision


def main():
    # Keep the held-out test partition untouched.
    X_train, _, y_train, _ = get_split()
    y = y_train.to_numpy()

    if y.sum() == 0:
        raise ValueError("Training target contains no positive attrition cases.")

    per_repeat_rows = []
    first_repeat_probabilities = {}

    for model_name, (classifier, scale_numeric) in get_models().items():
        pipeline = build_model_pipeline(
            classifier=classifier,
            scale_numeric=scale_numeric,
            use_smote=False,
        )

        for repeat in range(N_REPEATS):
            cv = StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=BASE_SEED + repeat,
            )

            # Every row receives an out-of-fold probability from a model
            # that was not fitted on that row.
            probabilities = cross_val_predict(
                clone(pipeline),
                X_train,
                y_train,
                cv=cv,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]

            row = {
                "model": model_name,
                "repeat": repeat + 1,
                "average_precision": average_precision_score(y, probabilities),
                "roc_auc": roc_auc_score(y, probabilities),
            }

            # A constant DummyClassifier cannot rank employees. Its top-k
            # metrics would depend on arbitrary row order, so report NaN.
            if model_name == "Dummy":
                for fraction in CAPACITY_FRACTIONS:
                    pct = int(fraction * 100)
                    row[f"recall_top{pct}pct"] = np.nan
                    row[f"precision_top{pct}pct"] = np.nan
            else:
                for fraction in CAPACITY_FRACTIONS:
                    recall, precision = top_k_metrics(y, probabilities, fraction)
                    pct = int(fraction * 100)
                    row[f"recall_top{pct}pct"] = recall
                    row[f"precision_top{pct}pct"] = precision

            per_repeat_rows.append(row)

            if repeat == 0:
                first_repeat_probabilities[model_name] = probabilities

        print(f"Completed: {model_name}")

    per_repeat = pd.DataFrame(per_repeat_rows)
    metric_columns = [
        column for column in per_repeat.columns
        if column not in ("model", "repeat")
    ]

    summary = per_repeat.groupby("model")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.sort_values("average_precision_mean", ascending=False)

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_repeat_path = Path("reports/model_comparison_repeats.csv")
    summary_path = Path("reports/model_comparison.csv")
    per_repeat.to_csv(per_repeat_path, index=False)
    summary.to_csv(summary_path, index=True, index_label="model")

    # PR curves from repeat 1 only; the other repeats are summarized in CSV.
    plt.figure(figsize=(8, 6))
    for model_name, probabilities in first_repeat_probabilities.items():
        if model_name == "Dummy":
            continue
        precision, recall, _ = precision_recall_curve(y, probabilities)
        ap = average_precision_score(y, probabilities)
        plt.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")

    prevalence = float(y.mean())
    plt.axhline(
        y=prevalence,
        linestyle="--",
        label=f"No-skill prevalence ({prevalence:.3f})",
    )
    plt.xlabel("Recall (attrition class)")
    plt.ylabel("Precision")
    plt.title("Out-of-fold precision-recall curves (training set, repeat 1)")
    plt.legend()
    plt.tight_layout()
    figure_path = figures_dir / "pr_curves_oof.png"
    plt.savefig(figure_path, dpi=160)
    plt.close()

    mean_columns = [
        column for column in summary.columns
        if column.endswith("_mean")
    ]
    print("\nMean over repeats (standard deviations are in the summary CSV):")
    print(summary[mean_columns].to_string(float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved: {per_repeat_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {figure_path}")
    print(
        "\nInterpretation note: standard deviations across the three repeats "
        "reflect partition variability, not full sampling uncertainty. "
        "Top-k metrics for the constant Dummy model are intentionally blank "
        "because it cannot rank employees."
    )
    print(
        "Test-set note: all comparison metrics use training-set out-of-fold "
        "predictions only; the held-out test set was not evaluated."
    )


if __name__ == "__main__":
    main()
