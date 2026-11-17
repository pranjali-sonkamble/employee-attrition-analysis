from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import get_split
from model_zoo import get_models
from preprocessing import build_model_pipeline


THRESHOLD = 0.50
N_BOOT = 2000
SEED = 42
FRACTIONS = [0.10, 0.20, 0.30]

EXPECTED = {
    "tn": 191,
    "fp": 56,
    "fn": 17,
    "tp": 30,
}


def point_metrics(y, p):
    pred = (p >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    out = {
        "average_precision": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "recall": tp / (tp + fn),
    }

    order = np.argsort(-p, kind="stable")

    for f in FRACTIONS:
        k = int(round(len(y) * f))
        out[
            f"recall_top{int(round(f * 100))}pct"
        ] = y[order[:k]].sum() / y.sum()

    return out, (
        int(tn),
        int(fp),
        int(fn),
        int(tp),
    )


def main():
    X_train, X_test, y_train, y_test = get_split()

    clf, scale = get_models()["Logistic Regression"]

    pipe = build_model_pipeline(
        clone(clf),
        scale_numeric=scale,
        use_smote=False,
    )

    pipe.fit(X_train, y_train)

    p = pipe.predict_proba(X_test)[:, 1]
    y = y_test.to_numpy()

    point, cm = point_metrics(y, p)

    got = dict(
        zip(
            ["tn", "fp", "fn", "tp"],
            cm,
        )
    )

    if got != EXPECTED:
        raise SystemExit(
            f"Does NOT reproduce the committed result: {got}"
        )

    print(
        "Reproduced committed confusion matrix:",
        got,
    )

    rng = np.random.default_rng(SEED)
    n = len(y)

    boots = []

    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)

        yb = y[idx]
        pb = p[idx]

        if yb.sum() == 0 or yb.sum() == n:
            continue

        boots.append(
            point_metrics(yb, pb)[0]
        )

    boot = pd.DataFrame(boots)

    rows = []

    for name, value in point.items():
        lo, hi = boot[name].quantile(
            [0.025, 0.975]
        )

        rows.append(
            {
                "metric": name,
                "estimate": value,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        "reports/test_uncertainty.csv",
        index=False,
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}",
        )
    )

    print(
        f"\nBootstrap: {len(boot)} valid resamples "
        f"of {n} test rows (seed={SEED}). "
        "Model and threshold were frozen before this run."
    )


if __name__ == "__main__":
    main()