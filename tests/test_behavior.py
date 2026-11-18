"""Behavior-level regression tests for the project's evaluation logic and outputs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_prep import get_split
from preprocessing import build_model_pipeline
from rule_baseline import rule_predictions
from segment_statistics import wilson_interval


def test_get_split_returns_disjoint_stratified_partitions():
    """Exercise the project's actual split function, not sklearn directly."""
    X_train, X_test, y_train, y_test = get_split()

    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert len(X_train) == 1176
    assert len(X_test) == 294

    overall_rate = pd.concat([y_train, y_test]).mean()
    assert abs(y_train.mean() - overall_rate) < 0.02
    assert abs(y_test.mean() - overall_rate) < 0.02


def test_pipeline_scaler_is_fitted_from_training_data_only():
    """Verify fitted preprocessing statistics come from X_train, not held-out X_test."""
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, _ = get_split()
    pipeline = build_model_pipeline(
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
        scale_numeric=True,
        use_smote=False,
    )
    pipeline.fit(X_train, y_train)

    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]

    numeric_columns = preprocessor.transformers_[0][2]
    train_imputed = pd.DataFrame(
        imputer.transform(X_train[numeric_columns]),
        columns=numeric_columns,
        index=X_train.index,
    )

    np.testing.assert_allclose(
        scaler.mean_,
        train_imputed.mean(axis=0).to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )

    test_imputed_mean = (
        pd.DataFrame(
            imputer.transform(X_test[numeric_columns]),
            columns=numeric_columns,
            index=X_test.index,
        )
        .mean(axis=0)
        .to_numpy()
    )
    assert not np.allclose(
        scaler.mean_,
        test_imputed_mean,
        rtol=1e-8,
        atol=1e-8,
    )


def test_final_metrics_match_committed_confusion_matrix():
    """Check that committed headline metrics agree with the confusion matrix."""
    metrics = pd.read_csv(ROOT / "reports" / "final_test_metrics.csv").iloc[0]
    matrix = pd.read_csv(
        ROOT / "reports" / "final_test_confusion_matrix.csv",
        index_col=0,
    ).to_numpy()

    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    assert int(metrics["test_rows"]) == total
    assert int(metrics["test_attrition_cases"]) == fn + tp
    assert np.isclose(metrics["precision"], precision)
    assert np.isclose(metrics["recall"], recall)
    assert np.isclose(metrics["f1"], f1)


def test_rule_baseline_requires_both_conditions():
    """Exercise the project's actual rule implementation."""
    X = pd.DataFrame({
        "OverTime": ["Yes", "Yes", "No", "No"],
        "YearsAtCompany": [2, 5, 2, 5],
    })
    assert rule_predictions(X).tolist() == [1, 0, 0, 0]


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(3, 11)
    observed = 3 / 11
    assert 0 <= low <= observed <= high <= 1


def test_base_rate_brier_equals_p_times_one_minus_p():
    """Exercise the project's calibration baseline calculation."""
    y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    observed_rate = y.mean()
    assert np.isclose(
        base_rate_brier_score(y),
        observed_rate * (1 - observed_rate),
    )


def test_apply_threshold_returns_expected_flags():
    """Exercise the project's actual thresholding function."""
    probabilities = np.array([0.10, 0.49, 0.50, 0.5001, 0.90])
    assert apply_threshold(probabilities, 0.50).tolist() == [0, 0, 1, 1, 1]
