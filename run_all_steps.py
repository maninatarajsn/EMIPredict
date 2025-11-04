"""
Run all EMIPredict AI pipeline steps in sequence:
1. Data Loading & Preprocessing
2. Feature Engineering
3. Classification Modeling
"""
import subprocess
import sys

scripts = [
    "EMIPredict_AI/notebooks/01_data_loading_preprocessing.py",
    "EMIPredict_AI/notebooks/03_feature_engineering.py",
    "EMIPredict_AI/notebooks/04_classification_modeling.py",
    "EMIPredict_AI/notebooks/05_regression_modeling.py"
]

for script in scripts:
    print(f"\n=== Running: {script} ===")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with exit code {result.returncode}")
        print(result.stderr)
        break
    else:
        print(f"[SUCCESS] {script} completed.")
