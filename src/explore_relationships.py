from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"

NUMERIC = [
    "Age",
    "DistanceFromHome",
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "JobLevel",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "DailyRate",
    "HourlyRate",
    "MonthlyRate",
]

ORDINAL = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "StockOptionLevel",
    "PerformanceRating",
]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)

    lines = ["RELATIONSHIP EDA", "=" * 50, ""]

    corr = df[NUMERIC].corr().round(2)
    lines += ["Feature correlation matrix (Pearson):", corr.to_string(), ""]

    pairs = (
        corr.where(corr.abs().lt(1.0))
        .abs()
        .unstack()
        .dropna()
        .sort_values(ascending=False)
    )

    seen, top = set(), []
    for (a, b), _ in pairs.items():
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            top.append((a, b, corr.loc[a, b]))

    lines.append("Top 10 absolute pairwise correlations:")
    for a, b, value in top[:10]:
        lines.append(f"{a} vs {b}: {value}")
    lines.append("")

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )
    plt.title("Correlation among numeric features")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    lines.append("Attrition rate by ordinal feature level:")
    for col in ORDINAL:
        summary = (
            df.groupby(col)["AttritionFlag"]
            .agg(records="size", attrition_rate_pct="mean")
        )
        summary["attrition_rate_pct"] = (
            summary["attrition_rate_pct"] * 100
        ).round(2)
        lines += [f"\n{col}:", summary.to_string()]

    (REPORT_DIR / "relationship_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("Done: reports/relationship_summary.txt")


if __name__ == "__main__":
    main()
