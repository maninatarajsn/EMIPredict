import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
import os

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

# Load feature-engineered data
feature_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/feature_engineered_data_regression.csv"
df = pd.read_csv(feature_path)

# Align features to match features_used.txt (as in classification modeling)
features_used_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/features_used_regression.txt"
if os.path.exists(features_used_path):
    with open(features_used_path, "r") as f:
        features_used = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(features_used)} features from features_used.txt")
else:
    features_used = list(df.columns)
    print("features_used.txt not found, using all columns.")

for col in features_used:
    if col not in df.columns:
        df[col] = 0
df = df[features_used]

df_target = pd.read_csv("/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/feature_engineered_data.csv")  # or from cleaned_data.csv

df['max_monthly_emi'] = df_target['max_monthly_emi']

X = df[features_used]
y = df['max_monthly_emi']

# Drop rows where target is NaN
nan_mask = ~y.isna()
X = X[nan_mask]
y = y[nan_mask]

# Drop non-numeric columns (after alignment)
drop_cols = X.select_dtypes(include=['object', 'category']).columns
if len(drop_cols) > 0:
    print(f"Dropping non-numeric columns from features: {list(drop_cols)}")
    X = X.drop(drop_cols, axis=1)
print(f"Final feature columns used for regression: {list(X.columns)}")

# Save feature order for inference
feature_order_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/regression_features_used.txt"
with open(feature_order_path, "w") as f:
    for col in X.columns:
        f.write(col + "\n")
print(f"✓ Regression feature order saved to: {feature_order_path}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = [
    (LinearRegression(), "LinearRegression"),
    (RandomForestRegressor(n_estimators=20, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1), "RandomForestRegressor"),
    (XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, objective='reg:squarederror'), "XGBoostRegressor"),
    (DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=42), "DecisionTreeRegressor")
]

results = []

for model, name in models:
    print(f"\n[{name}] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # For older scikit-learn, compute RMSE manually
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
    results.append({"name": name, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "model": model})
    # MLflow logging
    try:
        mlflow.set_experiment("emi_max_emi_regression")
        with mlflow.start_run(run_name=name):
            input_example = X_train.iloc[:2] if len(X_train) > 1 else X_train
            mlflow.sklearn.log_model(model, name, input_example=input_example)
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2, "mape": mape})
            print(f"✓ {name} model and metrics logged to MLflow.")
    except Exception as e:
        print(f"MLflow logging skipped for {name}: {e}")

# Select best model (lowest RMSE)
best = min(results, key=lambda x: x['rmse'])
print(f"\nBest model: {best['name']} (RMSE: {best['rmse']:.2f})")

# MLflow Model Registry integration (automatic)
try:
    mlflow.set_experiment("emi_max_emi_regression")
    with mlflow.start_run(run_name=f"Register_{best['name']}_BestModel") as run:
        input_example = X_train.iloc[:2] if len(X_train) > 1 else X_train
        print(f"Registering best model '{best['name']}' to MLflow Model Registry...")
        mlflow.sklearn.log_model(best['model'], best['name'], input_example=input_example, registered_model_name=f"BestEMIRegressor")
        print(f"\n✓ Best model '{best['name']}' registered as 'BestEMIRegressor' in MLflow Model Registry.")
        print("You can now view, version, and deploy this model from the MLflow UI under the 'Models' tab.")
except Exception as e:
    print(f"Model registry step skipped: {e}")
