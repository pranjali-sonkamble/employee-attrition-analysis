
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
TARGET = "Attrition"
DROP_COLS = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_data():
    return pd.read_csv(DATA_PATH)


def get_features_and_target(df):
    X = df.drop(columns=DROP_COLS + [TARGET])
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def get_split():
    X, y = get_features_and_target(load_data())
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_split()

    lines = [
        "TRAIN/TEST SPLIT SUMMARY",
        "=" * 50,
        f"test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratified=True",
        f"Dropped columns: {DROP_COLS}",
        f"Feature count: {X_train.shape[1]}",
        "",
        f"Train: {X_train.shape[0]} rows, "
        f"{int(y_train.sum())} leavers ({y_train.mean():.4f})",
        f"Test:  {X_test.shape[0]} rows, "
        f"{int(y_test.sum())} leavers ({y_test.mean():.4f})",
        "",
        f"Index overlap between train and test: "
        f"{len(set(X_train.index) & set(X_test.index))}",
        f"Missing values in train features: "
        f"{int(X_train.isna().sum().sum())}",
        f"Feature columns: {list(X_train.columns)}",
    ]

    Path("reports").mkdir(exist_ok=True)
    Path("reports/split_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("\n".join(lines))
