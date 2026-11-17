import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Allow the project modules in src/ to be imported when pytest runs from
# the repository root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_prep import (  # noqa: E402
    DROP_COLS,
    TARGET,
    get_features_and_target,
    get_split,
    load_data,
)
from preprocessing import (  # noqa: E402
    ALL_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    build_model_pipeline,
    build_preprocessor,
)


def test_dataset_loads_with_expected_shape():
    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1470, 35)
    assert TARGET in df.columns


def test_target_is_binary_and_drop_columns_are_removed():
    df = load_data()
    X, y = get_features_and_target(df)

    assert set(y.unique()) == {0, 1}
    assert y.dtype == np.int64

    for column in DROP_COLS:
        assert column not in X.columns

    assert TARGET not in X.columns


def test_expected_feature_count():
    df = load_data()
    X, _ = get_features_and_target(df)

    # 35 original columns
    # - 1 target
    # - 4 explicitly dropped columns
    # = 30 model input features
    assert X.shape[1] == 30


def test_split_is_reproducible_and_has_no_index_overlap():
    X_train_1, X_test_1, y_train_1, y_test_1 = get_split()
    X_train_2, X_test_2, y_train_2, y_test_2 = get_split()

    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)

    assert set(X_train_1.index).isdisjoint(set(X_test_1.index))


def test_split_has_expected_size_and_stratification():
    X_train, X_test, y_train, y_test = get_split()

    assert len(X_train) == 1176
    assert len(X_test) == 294

    overall_rate = pd.concat([y_train, y_test]).mean()

    assert abs(y_train.mean() - overall_rate) < 0.02
    assert abs(y_test.mean() - overall_rate) < 0.02


def test_preprocessor_contains_expected_feature_groups():
    preprocessor = build_preprocessor(scale_numeric=True)

    transformer_names = {
        name for name, _, _ in preprocessor.transformers
    }

    assert "numeric" in transformer_names
    assert "categorical" in transformer_names

    numeric_columns = set(ALL_NUMERIC_FEATURES)
    categorical_columns = set(CATEGORICAL_FEATURES)

    assert len(numeric_columns) == 23
    assert len(categorical_columns) == 7
    assert numeric_columns.isdisjoint(categorical_columns)


def test_preprocessor_produces_no_missing_values():
    X_train, _, _, _ = get_split()

    preprocessor = build_preprocessor(scale_numeric=True)
    transformed = preprocessor.fit_transform(X_train)

    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert dense.shape[0] == len(X_train)
    assert np.isfinite(dense).all()


def test_model_pipeline_has_expected_structure():
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
    )

    pipeline = build_model_pipeline(
        classifier=classifier,
        scale_numeric=True,
        use_smote=False,
    )

    assert list(pipeline.named_steps.keys()) == [
        "preprocessor",
        "classifier",
    ]

    assert pipeline.named_steps["classifier"].C == 1.0
    assert pipeline.named_steps["classifier"].class_weight == "balanced"


def test_model_pipeline_fits_and_predicts_probabilities():
    X_train, X_test, y_train, _ = get_split()

    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
    )

    pipeline = build_model_pipeline(
        classifier=classifier,
        scale_numeric=True,
        use_smote=False,
    )

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    assert len(probabilities) == len(X_test)
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_model_pipeline_produces_binary_predictions():
    X_train, X_test, y_train, _ = get_split()

    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
    )

    pipeline = build_model_pipeline(
        classifier=classifier,
        scale_numeric=True,
        use_smote=False,
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(np.unique(predictions)).issubset({0, 1})
