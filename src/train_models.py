
from pathlib import Path
import sys

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import get_split
from preprocessing import build_model_pipeline


RANDOM_STATE = 42
N_SPLITS = 5


def main():
    X_train, _, y_train, _ = get_split()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "Dummy": (
            DummyClassifier(strategy="most_frequent"),
            False,
        ),
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
        "Hist Gradient Boosting": (
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.1,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
    }

    scoring = {
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "average_precision": "average_precision",
    }

    results = []

    for name, (classifier, scale_numeric) in models.items():
        pipeline = build_model_pipeline(
            classifier=classifier,
            scale_numeric=scale_numeric,
            use_smote=False,
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )

        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        results.append(row)

        print(
            f"{name}: "
            f"Recall={row['recall_mean']:.3f}, "
            f"Precision={row['precision_mean']:.3f}, "
            f"F1={row['f1_mean']:.3f}, "
            f"Average Precision={row['average_precision_mean']:.3f}"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="average_precision_mean",
        ascending=False,
    )

    Path("reports").mkdir(exist_ok=True)
    output_path = Path("reports/baseline_cv_results.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved cross-validation results to {output_path}")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
