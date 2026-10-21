
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


def build_preprocessor(scale_numeric=True):
    """Build preprocessing that is fitted only on the training data."""
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
            ("numeric", numeric_pipeline, ALL_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model_pipeline(classifier, scale_numeric=True, use_smote=False):
    """Combine preprocessing, optional SMOTE, and a classifier."""
    steps = [
        ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
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
        "Test data used for fitting: No",
    ]

    from pathlib import Path

    Path("reports").mkdir(exist_ok=True)
    Path("reports/preprocessing_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("\n".join(lines))
