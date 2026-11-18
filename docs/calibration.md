# Calibration analysis

The frozen Logistic Regression is evaluated for probability calibration on the same held-out test partition used for the final model evaluation.

The script reports a 10-bin reliability curve and Brier score. It also computes the Brier score of a constant prediction equal to the observed test-set attrition rate, which provides a simple reference point for interpreting the model's Brier score. The calibration plot is descriptive: it checks whether predicted probabilities broadly track observed attrition frequencies. It does not alter the frozen model or threshold.

## Result

On the 294-row held-out test set, the Logistic Regression has a **Brier score of 0.1561**. The observed attrition rate is **47 / 294 = 0.1599**, giving a constant base-rate Brier score of **0.1343**.

Because the model uses `class_weight="balanced"`, its raw `predict_proba` outputs are not calibrated to be interpreted as real-world attrition probabilities. In this evaluation, the model's Brier score is worse than the simple constant-base-rate reference. Therefore, the probabilities should be treated as **ranking/screening scores rather than calibrated individual risk estimates**. No recalibration was applied.

This is a diagnostic result from one held-out split, not evidence that the model could never be calibrated. A production system would require calibration on appropriate validation data and monitoring after deployment.

## Reproduce

    python src/calibration.py

Outputs:

- `reports/figures/calibration_curve.png`
- `reports/calibration_metrics.csv`

The CSV includes `brier_score`, `base_rate`, and `base_rate_brier` so the conclusion is reproducible from the generated artifact.

Because the test set contains 294 employees and 47 attrition cases, calibration should be interpreted as a diagnostic rather than a production-grade probability-calibration guarantee.
