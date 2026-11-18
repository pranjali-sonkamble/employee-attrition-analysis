"""Generate a reliability/calibration plot for the frozen Logistic Regression."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import get_split
from model_zoo import get_models
from preprocessing import build_model_pipeline


def base_rate_brier_score(y_true):
    """Return the Brier score for a constant prediction equal to the observed base rate."""
    y = np.asarray(y_true)
    base_rate = float(y.mean())
    return brier_score_loss(y, np.full(len(y), base_rate))


def main():
    X_train, X_test, y_train, y_test = get_split()
    classifier, scale_numeric = get_models()["Logistic Regression"]

    pipeline = build_model_pipeline(
        classifier=classifier,
        scale_numeric=scale_numeric,
        use_smote=False,
    )
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    brier = brier_score_loss(y_test, probabilities)
    base_rate = float(y_test.mean())
    base_rate_brier = base_rate_brier_score(y_test)

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    CalibrationDisplay.from_predictions(
        y_test,
        probabilities,
        n_bins=10,
        strategy="uniform",
        name="Logistic Regression",
        ax=ax,
    )
    ax.set_title("Calibration curve — held-out test set")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed attrition rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output = Path("reports/figures/calibration_curve.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)

    metrics = pd.DataFrame([{
        "model": "Logistic Regression",
        "test_rows": len(y_test),
        "brier_score": brier,
        "base_rate": base_rate,
        "base_rate_brier": base_rate_brier,
        "n_bins": 10,
        "strategy": "uniform",
    }])
    metrics.to_csv("reports/calibration_metrics.csv", index=False)

    print("Calibration analysis")
    print("====================")
    print(f"Test rows: {len(y_test)}")
    print(f"Brier score: {brier:.4f}")
    print(f"Observed base rate: {base_rate:.4f}")
    print(f"Constant base-rate Brier score: {base_rate_brier:.4f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
