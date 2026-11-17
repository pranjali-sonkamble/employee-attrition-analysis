"""Add uncertainty and association tests for the project's main segments.

Rates use Wilson 95% intervals. Chi-square tests compare each main segment
with its complement on observed attrition status. These are association tests,
not causal tests.
"""

from math import sqrt
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

from data_prep import load_data

Z = 1.959963984540054


def wilson_interval(successes, total, z=Z):
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total == 0:
        return float("nan"), float("nan")

    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def segment_rows(df):
    attrition = (df["Attrition"] == "Yes").astype(int)

    definitions = {
        "Overtime": df["OverTime"].eq("Yes"),
        "Sales Representative": df["JobRole"].eq("Sales Representative"),
        "Early tenure (<=3 years)": df["YearsAtCompany"].le(3),
    }

    rows = []
    for name, mask in definitions.items():
        mask = mask.fillna(False)
        n = int(mask.sum())
        cases = int(attrition[mask].sum())
        rate = cases / n if n else float("nan")
        low, high = wilson_interval(cases, n)

        table = pd.crosstab(mask, attrition).reindex(
            index=[False, True],
            columns=[0, 1],
            fill_value=0,
        )
        chi2, p_value, dof, expected = chi2_contingency(table)

        rows.append({
            "segment": name,
            "rows": n,
            "attrition_cases": cases,
            "attrition_rate": rate,
            "rate_ci95_low": low,
            "rate_ci95_high": high,
            "chi2": chi2,
            "chi2_dof": dof,
            "chi2_p_value": p_value,
            "expected_min": float(expected.min()),
        })

    return pd.DataFrame(rows)


def main():
    df = load_data()
    result = segment_rows(df)

    Path("reports").mkdir(exist_ok=True)
    output = Path("reports/main_segment_statistics.csv")
    result.to_csv(output, index=False)

    print("Main segment uncertainty and association tests")
    print("==============================================")
    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print("\nInterpretation:")
    print(
        "Wilson intervals quantify uncertainty around descriptive segment "
        "rates. Chi-square p-values test association between segment "
        "membership and observed attrition; they do not establish causality."
    )
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
