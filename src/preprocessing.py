
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from data_prep import get_split


CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# Keep the coded ordinal variables as numeric values.
ORDINAL_FEATURES = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "StockOptionLevel",
    "PerformanceRating",
]

NUMERIC_FEATURES = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "JobLevel",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES


def build_preprocessor(scale_numeric=True, excluded_features=None):
    """Build preprocessing that is fitted only on the training data.

    ``excluded_features`` removes named columns from the modeling design while
    leaving the source dataframe unchanged for evaluation and subgroup analysis.
    """
    excluded_features = set(excluded_features or [])
    numeric_features = [
        feature for feature in ALL_NUMERIC_FEATURES
        if feature not in excluded_features
    ]
    categorical_features = [
        feature for feature in CATEGORICAL_FEATURES
        if feature not in excluded_features
    ]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = SklearnPipeline(numeric_steps)

    categorical_pipeline = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_model_pipeline(
    classifier,
    scale_numeric=True,
    use_smote=False,
    excluded_features=None,
):
    """Combine leakage-aware preprocessing, optional SMOTE, and a classifier."""
    steps = [
        (
            "preprocessor",
            build_preprocessor(
                scale_numeric=scale_numeric,
                excluded_features=excluded_features,
            ),
        ),
    ]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))

    steps.append(("classifier", classifier))
    return ImbPipeline(steps)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_split()

    preprocessor = build_preprocessor(scale_numeric=True)
    X_train_transformed = preprocessor.fit_transform(X_train)

    lines = [
        "PREPROCESSING VALIDATION",
        "=" * 50,
        f"Training rows: {X_train.shape[0]}",
        f"Test rows: {X_test.shape[0]}",
        f"Input features: {X_train.shape[1]}",
        f"Transformed training shape: {X_train_transformed.shape}",
        f"Categorical features: {len(CATEGORICAL_FEATURES)}",
        f"Ordinal features: {len(ORDINAL_FEATURES)}",
        f"Other numeric features: {len(NUMERIC_FEATURES)}",
        f"Missing values in transformed training data: "
        f"{int(__import__('numpy').isnan(X_train_transformed).sum())}",
        "Preprocessor fitted on training rows only (by construction)",
    ]

    from pathlib import Path

    Path("reports").mkdir(exist_ok=True)
    Path("reports/preprocessing_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("\n".join(lines))
