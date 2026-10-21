
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"

CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

NUMERIC_FEATURES = [
    "Age",
    "DistanceFromHome",
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsAtCompany",
]

sns.set_theme(style="whitegrid")


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)

    report_lines = [
        "EXPLORATORY DATA ANALYSIS",
        "=" * 60,
        f"Dataset shape: {df.shape[0]} rows x {df.shape[1] - 1} original columns",
        "",
        "Attrition counts:",
        df["Attrition"].value_counts().to_string(),
        "",
        "Attrition proportions:",
        df["Attrition"].value_counts(normalize=True).round(4).to_string(),
        "",
    ]

    # Overall target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Attrition", order=["No", "Yes"])
    plt.title("Employee Attrition Distribution")
    plt.xlabel("Attrition")
    plt.ylabel("Number of records")
    save_figure("01_attrition_distribution.png")

    # Attrition rate by categorical feature
    categorical_summaries = []

    for feature in CATEGORICAL_FEATURES:
        summary = (
            df.groupby(feature, observed=True)["AttritionFlag"]
            .agg(records="size", attrition_rate="mean")
            .reset_index()
        )
        summary["attrition_rate_pct"] = summary["attrition_rate"] * 100
        categorical_summaries.append(summary.assign(feature=feature))

        report_lines.extend([
            f"Attrition by {feature}:",
            summary[
                [feature, "records", "attrition_rate_pct"]
            ].round(2).to_string(index=False),
            "",
        ])

        plot_data = summary.sort_values("attrition_rate")

        plt.figure(figsize=(max(7, len(plot_data) * 1.1), 5))
        sns.barplot(
            data=plot_data,
            x=feature,
            y="attrition_rate_pct",
            color="steelblue",
        )
        plt.title(f"Attrition Rate by {feature}")
        plt.xlabel(feature)
        plt.ylabel("Attrition rate (%)")
        plt.xticks(rotation=35, ha="right")
        save_figure(f"attrition_by_{feature.lower()}.png")

    if categorical_summaries:
        pd.concat(categorical_summaries, ignore_index=True).to_csv(
            REPORT_DIR / "categorical_attrition_rates.csv",
            index=False,
        )

    # Attrition rates across numeric feature bands
    numeric_summaries = []

    for feature in NUMERIC_FEATURES:
        band = pd.qcut(df[feature], q=4, duplicates="drop")
        summary = (
            df.assign(FeatureBand=band)
            .groupby("FeatureBand", observed=True)["AttritionFlag"]
            .agg(records="size", attrition_rate="mean")
            .reset_index()
        )
        summary["feature"] = feature
        summary["feature_band"] = summary["FeatureBand"].astype(str)
        summary["attrition_rate_pct"] = summary["attrition_rate"] * 100
        numeric_summaries.append(
            summary[
                ["feature", "feature_band", "records",
                 "attrition_rate", "attrition_rate_pct"]
            ]
        )

        report_lines.extend([
            f"Attrition by {feature} bands:",
            summary[
                ["feature_band", "records", "attrition_rate_pct"]
            ].round(2).to_string(index=False),
            "",
        ])

        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=summary,
            x="feature_band",
            y="attrition_rate_pct",
            color="darkorange",
        )
        plt.title(f"Attrition Rate by {feature} Band")
        plt.xlabel(feature)
        plt.ylabel("Attrition rate (%)")
        plt.xticks(rotation=25, ha="right")
        save_figure(f"attrition_by_{feature.lower()}_bands.png")

    if numeric_summaries:
        pd.concat(numeric_summaries, ignore_index=True).to_csv(
            REPORT_DIR / "numeric_attrition_rates.csv",
            index=False,
        )

    (REPORT_DIR / "eda_summary.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("EDA completed.")
    print(f"Summary: {REPORT_DIR / 'eda_summary.txt'}")
    print(f"Categorical table: {REPORT_DIR / 'categorical_attrition_rates.csv'}")
    print(f"Numeric table: {REPORT_DIR / 'numeric_attrition_rates.csv'}")
    print(f"Figures saved in: {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()
