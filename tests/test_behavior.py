"""Behavior-level regression tests for the evaluation logic."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rule_baseline import rule_predictions
from segment_statistics import wilson_interval


def test_split_is_stratified():
    y = pd.Series([0] * 84 + [1] * 16)
    indices = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    train_rate = y.iloc[train_idx].mean()
    test_rate = y.iloc[test_idx].mean()

    assert abs(train_rate - test_rate) <= 0.01


def test_train_test_ids_do_not_overlap():
    train_ids = {101, 102, 103, 104}
    test_ids = {105, 106}
    assert train_ids.isdisjoint(test_ids)


def test_threshold_is_applied_at_0_50():
    probabilities = np.array([0.49, 0.50, 0.5001, 0.90])
    predictions = (probabilities >= 0.50).astype(int)
    assert predictions.tolist() == [0, 1, 1, 1]


def test_confusion_metrics_are_correct():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    assert (tn, fp, fn, tp) == (2, 1, 1, 2)
    assert precision_score(y_true, y_pred) == 2 / 3
    assert recall_score(y_true, y_pred) == 2 / 3


def test_rule_baseline_requires_both_conditions():
    X = pd.DataFrame({
        "OverTime": ["Yes", "Yes", "No", "No"],
        "YearsAtCompany": [2, 5, 2, 5],
    })
    assert rule_predictions(X).tolist() == [1, 0, 0, 0]


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(3, 11)
    observed = 3 / 11
    assert 0 <= low <= observed <= high <= 1
