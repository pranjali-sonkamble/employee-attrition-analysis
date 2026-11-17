from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import get_split
from model_zoo import get_models
from preprocessing import build_model_pipeline


OUTPUT_CSV = Path("reports/model_interpretation.csv")
OUTPUT_FIGURE = Path("reports/figures/model_coefficients.png")
TOP_N = 20


def main():
    X_train, _, y_train, _ = get_split()

    clf, scale = get_models()["Logistic Regression"]

    pipeline = build_model_pipeline(
        clone(clf),
        scale_numeric=scale,
        use_smote=False,
    )

    pipeline.fit(X_train, y_train)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]

    interpretation = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "absolute_coefficient": np.abs(coefficients),
            "direction": np.where(
                coefficients > 0,
                "Higher model log-odds of attrition",
                "Lower model log-odds of attrition",
            ),
        }
    ).sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    interpretation.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    top = interpretation.head(TOP_N).sort_values(
        "coefficient"
    )

    plt.figure(figsize=(10, 8))
    plt.barh(
        top["feature"],
        top["coefficient"],
    )
    plt.axvline(
        0,
        linewidth=1,
    )
    plt.xlabel("Logistic Regression coefficient")
    plt.ylabel("Encoded feature")
    plt.title(
        "Top 20 Logistic Regression coefficients"
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FIGURE,
        dpi=200,
    )
    plt.close()

    print("Model interpretation")
    print("====================")
    print(f"Training rows used: {len(X_train)}")
    print(
        f"Transformed feature count: {len(feature_names)}"
    )

    print("\nTop positive coefficients:")
    print(
        interpretation[
            interpretation["coefficient"] > 0
        ]
        .head(TOP_N // 2)
        [
            [
                "feature",
                "coefficient",
                "odds_ratio",
            ]
        ]
        .to_string(index=False)
    )

    print("\nTop negative coefficients:")
    print(
        interpretation[
            interpretation["coefficient"] < 0
        ]
        .sort_values(
            "coefficient",
            ascending=True,
        )
        .head(TOP_N // 2)
        [
            [
                "feature",
                "coefficient",
                "odds_ratio",
            ]
        ]
        .to_string(index=False)
    )

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()