"""Run the main reproducible analysis stages in sequence."""

import subprocess
import sys


STAGES = [
    "src/inspect_data.py",
    "src/validate_columns.py",
    "src/explore_data.py",
    "src/explore_relationships.py",
    "src/compare_models.py",
    "src/evaluate_thresholds.py",
    "src/final_evaluation.py",
    "src/test_uncertainty.py",
    "src/model_interpretation.py",
    "src/subgroup_evaluation.py",
    "src/sensitive_attribute_ablation.py",
]


def main():
    for script in STAGES:
        print(f"\n{'=' * 72}\nRunning {script}\n{'=' * 72}")
        subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    main()
