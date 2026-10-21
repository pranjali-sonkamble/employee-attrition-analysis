
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH.resolve()}. "
            "Check the filename and location."
        )

    df = pd.read_csv(DATA_PATH)

    print("DATASET INSPECTION")
    print("=" * 50)

    print("\nShape (rows, columns):")
    print(df.shape)

    print("\nColumn data types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "None")

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nConstant columns (one unique value):")
    print(df.columns[df.nunique(dropna=False) == 1].tolist())

    print("\nTarget distribution: Attrition")
    if "Attrition" in df.columns:
        print("\nCounts:")
        print(df["Attrition"].value_counts(dropna=False).to_string())

        print("\nProportions:")
        print(
            df["Attrition"]
            .value_counts(normalize=True, dropna=False)
            .round(4)
            .to_string()
        )
    else:
        print("ERROR: Attrition column not found.")

    print("\nUnique values in categorical columns:")
    for col in df.select_dtypes(include=["object", "category"]).columns:
        values = df[col].dropna().unique().tolist()
        print(f"{col}: {values}")


if __name__ == "__main__":
    main()
