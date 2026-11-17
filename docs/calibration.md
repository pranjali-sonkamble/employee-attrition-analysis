# Calibration analysis

The frozen Logistic Regression is evaluated for probability calibration on the same held-out test partition used for the final model evaluation.

The script reports a 10-bin reliability curve and Brier score. The calibration plot is descriptive: it checks whether predicted probabilities broadly track observed attrition frequencies. It does not alter the frozen model or threshold.

Reproduce with:

    python src/calibration.py

Outputs:

- `reports/figures/calibration_curve.png`
- `reports/calibration_metrics.csv`

Because the test set contains 294 employees and 47 attrition cases, calibration should be interpreted as a diagnostic rather than a production-grade probability-calibration guarantee.
