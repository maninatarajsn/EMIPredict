"""
Step 4: Classification Modeling
- Train and evaluate classification models for EMI eligibility prediction
- Mandatory: Logistic Regression, Random Forest, XGBoost
- Optional: Decision Tree
- Track experiments with MLflow (if available)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow
import mlflow.sklearn
import os
from xgboost import XGBClassifier

# === EMI Eligibility Model Debug & Retrain Checklist ===
# 1. Check class balance in your target variable
# 2. Review confusion matrix and classification report after training
# 3. Use class_weight='balanced' or oversample ineligible cases if needed
# 4. Ensure feature engineering is identical in both training and inference
# 5. Check for data leakage (no target info in features)
# 6. Tune model hyperparameters for better discrimination
# 7. Use probability threshold tuning in deployment
# 8. Validate with realistic, high-risk test cases

# Load feature-engineered data
df = pd.read_csv("/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/feature_engineered_data_classification.csv")

# Target and features
# Use the correct target column from feature engineering output
if 'emi_eligibility_Eligible' in df.columns:
    target_col = 'emi_eligibility_Eligible'
elif 'emi_eligibility' in df.columns:
    target_col = 'emi_eligibility'
else:
    raise ValueError("Target column for EMI eligibility not found in data.")

# Only use numeric columns for modeling
X = df.drop([target_col], axis=1)
non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
if len(non_numeric_cols) > 0:
    print(f"Dropping non-numeric columns from features: {list(non_numeric_cols)}")
    X = X.drop(non_numeric_cols, axis=1)
y = df[target_col]

# 1. Check class balance
print("\n=== Target Class Balance ===")
print(y.value_counts())

# If target is not binary, encode it (should not be needed if feature engineering is correct)
if y.dtype == 'O' or y.nunique() > 2:
    y = y.map({'Eligible': 1, 'Not_Eligible': 0, 'Yes': 1, 'No': 0})

# Drop rows where target is NaN (to avoid ValueError in train_test_split)
from imblearn.over_sampling import RandomOverSampler

nan_mask = ~y.isna()
X = X[nan_mask]
y = y[nan_mask]

# Optional: Upsample minority class using RandomOverSampler
do_upsample = True  # Set to False to disable upsampling
if do_upsample:
    print("\nApplying RandomOverSampler to balance classes...")
    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_resample(X, y)
    print("Class balance after upsampling:")
    print(pd.Series(y).value_counts())

from sklearn.model_selection import train_test_split
# Use all data, stratified split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Validation size: {len(X_val)}")
print("Train class balance:\n", y_train.value_counts())
print("Test class balance:\n", y_test.value_counts())
print("Validation class balance:\n", y_val.value_counts())

# Save feature column order for inference alignment

# Align features to features_used.txt (do not overwrite it)
features_used_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/features_used_classification.txt"
if os.path.exists(features_used_path):
    with open(features_used_path, "r") as f:
        required_features = [line.strip() for line in f.readlines() if line.strip()]
    # Add missing features as 0, remove extras
    for col in required_features:
        if col not in X.columns:
            X[col] = 0
    X = X[required_features]
    print(f"✓ Features aligned to features_used.txt ({len(required_features)} features)")
else:
    print("Warning: features_used.txt not found. Using all columns.")

# 2. Use class_weight='balanced' for all models

def run_and_log_model(model, model_name, params=None):
    print(f"\n[{model_name}] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n=== {model_name} Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    # MLflow Tracking
    try:
        mlflow.set_experiment("emi_eligibility_classification")
        with mlflow.start_run(run_name=model_name):
            # Provide input_example for model signature
            input_example = X_train.iloc[:2] if len(X_train) > 1 else X_train
            mlflow.sklearn.log_model(model, model_name, input_example=input_example)
            if params:
                mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            print(f"\n✓ {model_name} model and metrics logged to MLflow.")
    except Exception as e:
        print(f"MLflow logging skipped for {model_name}: {e}")

# Train and evaluate all models, collect results
model_results = []

# 1. Logistic Regression (Baseline)
logreg_params = {"solver": "liblinear", "random_state": 42, "max_iter": 200, "class_weight": "balanced"}
logreg = LogisticRegression(**logreg_params)
model_results.append(("LogisticRegression", logreg, logreg_params, run_and_log_model(logreg, "LogisticRegression", logreg_params)))

# 2. Random Forest Classifier
rf_params = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 10, "random_state": 42, "class_weight": "balanced"}
rf = RandomForestClassifier(**rf_params)
model_results.append(("RandomForestClassifier", rf, rf_params, run_and_log_model(rf, "RandomForestClassifier", rf_params)))

# 3. XGBoost Classifier (required)
if 0 in y_train.value_counts() and 1 in y_train.value_counts():
    scale_pos_weight = float(y_train.value_counts()[0]) / float(y_train.value_counts()[1]) if y_train.value_counts()[1] > 0 else 1.0
else:
    scale_pos_weight = 1.0
xgb_params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42, "use_label_encoder": False, "eval_metric": "logloss", "scale_pos_weight": scale_pos_weight}
xgb = XGBClassifier(**xgb_params)
model_results.append(("XGBoostClassifier", xgb, xgb_params, run_and_log_model(xgb, "XGBoostClassifier", xgb_params)))

# 4. Decision Tree (Optional)
try:
    from sklearn.tree import DecisionTreeClassifier
    dt_params = {"random_state": 42, "max_depth": 6, "min_samples_leaf": 20}
    dt = DecisionTreeClassifier(**dt_params)
    model_results.append(("DecisionTreeClassifier", dt, dt_params, run_and_log_model(dt, "DecisionTreeClassifier", dt_params)))
except ImportError:
    print("DecisionTreeClassifier not available. Skipping.")

# 4. Review confusion matrix and classification report for all models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
best_name = None
best_model = None
best_acc = -1
for name, model, params, _ in model_results:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model {name} accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_model = model
print(f"\nBest classification model: {best_name} (Accuracy: {best_acc:.4f})")

# MLflow Model Registry integration (automatic)
try:
    mlflow.set_experiment("emi_eligibility_classification")
    with mlflow.start_run(run_name=f"Register_{best_name}_BestModel") as run:
        input_example = X_train.iloc[:2] if len(X_train) > 1 else X_train
        print(f"Registering best model '{best_name}' to MLflow Model Registry...")
        mlflow.sklearn.log_model(best_model, best_name, input_example=input_example, registered_model_name=f"BestEMIClassifier")
        print(f"\n✓ Best model '{best_name}' registered as 'BestEMIClassifier' in MLflow Model Registry.")
        print("You can now view, version, and deploy this model from the MLflow UI under the 'Models' tab.")

        # Set the latest version as 'production' automatically
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        versions = client.get_latest_versions("BestEMIClassifier", stages=["None", "Staging", "Production", "Archived"])
        if versions:
            latest_version = max([int(v.version) for v in versions])
            client.transition_model_version_stage(
                name="BestEMIClassifier",
                version=str(latest_version),
                stage="production",
                archive_existing_versions=True
            )
            print(f"\n✓ Model version {latest_version} transitioned to 'Production' stage.")
except Exception as e:
    print(f"Model registry step skipped: {e}")