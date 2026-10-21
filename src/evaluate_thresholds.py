
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import get_split
from preprocessing import build_model_pipeline


RANDOM_STATE = 42
N_SPLITS = 5

# Candidate thresholds to compare using training-set OOF predictions.
THRESHOLDS = np.arange(0.10, 0.61, 0.05)


def main():
    X_train, _, y_train, _ = get_split()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
    }

    results = []

    for name, (classifier, scale_numeric) in models.items():
        pipeline = build_model_pipeline(
            classifier=classifier,
            scale_numeric=scale_numeric,
            use_smote=False,
        )

        # Each training row receives a probability from a model
        # that did not train on that row.
        oof_probabilities = cross_val_predict(
            clone(pipeline),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        print(f"\n{name}")
        print("-" * len(name))
        print(
            "OOF average precision: "
            f"{average_precision_score(y_train, oof_probabilities):.3f}"
        )

        for threshold in THRESHOLDS:
            predictions = (oof_probabilities >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_train,
                predictions,
                labels=[0, 1],
            ).ravel()

            results.append({
                "model": name,
                "threshold": round(float(threshold), 2),
                "precision": precision_score(
                    y_train, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_train, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_train, predictions, zero_division=0
                ),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "flagged_employees": int(predictions.sum()),
            })

    results_df = pd.DataFrame(results)

    output_path = Path("reports/threshold_evaluation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved all threshold results to {output_path}")

    for name in models:
        model_results = results_df[
            results_df["model"] == name
        ].sort_values(
            by=["recall", "precision"],
            ascending=[False, False],
        )

        print(f"\nTop recall results: {name}")
        print(
            model_results.head(5).to_string(
                index=False,
                float_format=lambda value: f"{value:.3f}",
            )
        )

    print(
        "\nImportant: thresholds were evaluated using training-set "
        "out-of-fold predictions only. The held-out test set was not used."
    )


if __name__ == "__main__":
    main()
