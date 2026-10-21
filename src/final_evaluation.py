
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from data_prep import get_split
from model_zoo import get_models
from preprocessing import build_model_pipeline


def main():
    X_train, X_test, y_train, y_test = get_split()

    model, scale_numeric = get_models()["Logistic Regression"]
    pipeline = build_model_pipeline(
        clone(model),
        scale_numeric=scale_numeric,
        use_smote=False,
    )

    # Fit once on training data; evaluate once on held-out test data.
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Fixed threshold from the existing baseline evaluation.
    threshold = 0.50
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "model": "Logistic Regression",
        "threshold": threshold,
        "test_rows": len(y_test),
        "test_attrition_cases": int(y_test.sum()),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    figures = reports / "figures"
    figures.mkdir(exist_ok=True)

    pd.DataFrame([metrics]).to_csv(
        reports / "final_test_metrics.csv", index=False
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual No Attrition", "Actual Attrition"],
        columns=["Predicted No Attrition", "Predicted Attrition"],
    )
    matrix_df.to_csv(reports / "final_test_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No Attrition", "Attrition"],
        yticklabels=["No Attrition", "Attrition"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Logistic Regression — Held-out Test Set",
    )
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]),
                    ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Employees")
    fig.tight_layout()
    fig.savefig(figures / "final_test_confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("Final held-out test evaluation")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(matrix_df)
    print("\nSaved reports/final_test_metrics.csv")
    print("Saved reports/final_test_confusion_matrix.csv")
    print("Saved reports/figures/final_test_confusion_matrix.png")
    print("The test set was not used for model or threshold selection.")


if __name__ == "__main__":
    main()
