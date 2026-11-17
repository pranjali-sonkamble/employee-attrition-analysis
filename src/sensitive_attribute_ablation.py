"""Evaluate a frozen-model ablation that excludes sensitive attributes.

The baseline and ablated models use the same stratified train/test split,
classifier configuration, preprocessing, and 0.50 threshold. The ablated model
removes Age, Gender, and MaritalStatus from model inputs only. The original
columns remain available for subgroup grouping and are never used as model
features in the ablated pipeline.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import get_split
from model_zoo import get_models
from preprocessing import build_model_pipeline


THRESHOLD = 0.50
EXCLUDED_FEATURES = ["Age", "Gender", "MaritalStatus"]
OUTPUT_DIR = Path("reports")


def threshold_metrics(y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "average_precision": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "flag_rate": float(predictions.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def subgroup_metrics(y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    negatives = tn + fp
    return {
        "rows": len(y_true),
        "attrition_cases": int(y_true.sum()),
        "attrition_rate": float(y_true.mean()),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "false_positive_rate": (
            float(fp / negatives) if negatives else np.nan
        ),
        "flag_rate": float(predictions.mean()),
    }


def make_age_groups(age):
    return pd.cut(
        age,
        bins=[0, 30, 36, 43, np.inf],
        labels=["18-30", "31-36", "37-43", "44+"],
        right=True,
    )


def main():
    X_train, X_test, y_train, y_test = get_split()

    classifier, scale_numeric = get_models()["Logistic Regression"]

    baseline_pipeline = build_model_pipeline(
        clone(classifier),
        scale_numeric=scale_numeric,
        use_smote=False,
    )
    ablated_pipeline = build_model_pipeline(
        clone(classifier),
        scale_numeric=scale_numeric,
        use_smote=False,
        excluded_features=EXCLUDED_FEATURES,
    )

    baseline_pipeline.fit(X_train, y_train)
    ablated_pipeline.fit(
        X_train.drop(columns=EXCLUDED_FEATURES),
        y_train,
    )

    baseline_probabilities = baseline_pipeline.predict_proba(X_test)[:, 1]
    ablated_probabilities = ablated_pipeline.predict_proba(
        X_test.drop(columns=EXCLUDED_FEATURES)
    )[:, 1]

    summary_rows = []
    for model_name, probabilities in [
        ("Full model", baseline_probabilities),
        ("Without Age/Gender/MaritalStatus", ablated_probabilities),
    ]:
        metrics = threshold_metrics(y_test, probabilities)
        metrics["model"] = model_name
        metrics["excluded_features"] = (
            "None" if model_name == "Full model"
            else ", ".join(EXCLUDED_FEATURES)
        )
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)[[
        "model",
        "excluded_features",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "flag_rate",
        "tn",
        "fp",
        "fn",
        "tp",
    ]]

    subgroup_rows = []
    subgroup_definitions = [
        ("Gender", X_test["Gender"]),
        ("MaritalStatus", X_test["MaritalStatus"]),
        ("Age group", make_age_groups(X_test["Age"])),
    ]

    for subgroup_name, values in subgroup_definitions:
        for subgroup_value in sorted(values.dropna().unique()):
            mask = values == subgroup_value

            for model_name, probabilities in [
                ("Full model", baseline_probabilities),
                ("Without Age/Gender/MaritalStatus", ablated_probabilities),
            ]:
                row = subgroup_metrics(
                    y_test.loc[mask],
                    probabilities[mask],
                )
                row.update({
                    "model": model_name,
                    "group": subgroup_name,
                    "subgroup": str(subgroup_value),
                })
                subgroup_rows.append(row)

    subgroup = pd.DataFrame(subgroup_rows)[[
        "model",
        "group",
        "subgroup",
        "rows",
        "attrition_cases",
        "attrition_rate",
        "precision",
        "recall",
        "false_positive_rate",
        "flag_rate",
    ]]

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_path = OUTPUT_DIR / "sensitive_attribute_ablation.csv"
    subgroup_path = OUTPUT_DIR / "sensitive_attribute_ablation_subgroups.csv"
    summary.to_csv(summary_path, index=False)
    subgroup.to_csv(subgroup_path, index=False)

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    report_lines = [
        "# Sensitive-Attribute Ablation",
        "",
        "## Purpose",
        "",
        "Compare the frozen Logistic Regression specification with an otherwise identical model that excludes Age, Gender, and MaritalStatus from model inputs.",
        "",
        "The same stratified train/test split, preprocessing approach, classifier configuration, random seed, and 0.50 threshold are used for both models. The three attributes remain available only for subgroup grouping in the ablated evaluation.",
        "",
        "## Results",
        "",
        "Run python src/sensitive_attribute_ablation.py to regenerate the tables below. The committed CSV files are the source of the reported values.",
        "",
        "### Overall test-set metrics",
        "",
        "| Model | Average Precision | ROC-AUC | Precision | Recall | Flag rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in summary.itertuples(index=False):
        report_lines.append(
            f"| {row.model} | {row.average_precision:.4f} | "
            f"{row.roc_auc:.4f} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.flag_rate:.2%} |"
        )

    report_lines += [
        "",
        "### Subgroup results",
        "",
        "The subgroup table compares the same fixed threshold within Gender, MaritalStatus, and the project's predefined age bands. These are descriptive diagnostics, not a fairness certification.",
        "",
        "See reports/sensitive_attribute_ablation_subgroups.csv for the complete results.",
        "",
        "## Interpretation",
        "",
        "If the ablated model changes predictive performance or subgroup metrics materially, that difference should be treated as evidence that these attributes contribute predictive information in this benchmark. It does not establish causality, fairness, or what would happen in another organization.",
        "",
        "Because the dataset is a fictional benchmark and the test partition contains only 47 attrition cases, small subgroup differences should not be over-interpreted.",
        "",
        "## Responsible-use note",
        "",
        "The ablation is an analytical diagnostic. It does not establish that excluding these attributes would make the model appropriate for employment decisions. Any real HR application would require organization-specific validation, privacy safeguards, a predefined fairness framework, governance, and human oversight.",
    ]

    (docs_dir / "sensitive_attribute_ablation.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("Sensitive-attribute ablation")
    print("===========================")
    print(f"Excluded model features: {', '.join(EXCLUDED_FEATURES)}")
    print(f"Threshold: {THRESHOLD}")
    print("\nOverall test-set comparison:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:")
    print(f"- {summary_path}")
    print(f"- {subgroup_path}")
    print("- docs/sensitive_attribute_ablation.md")


if __name__ == "__main__":
    main()
