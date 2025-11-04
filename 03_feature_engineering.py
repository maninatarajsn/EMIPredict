"""
Step 3: Feature Engineering
- Create derived financial ratios (debt-to-income, expense-to-income, affordability ratios)
- Generate risk scoring features based on credit history and employment stability
- Apply categorical encoding and numerical feature scaling
- Develop interaction features between key financial variables
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Load cleaned data
df = pd.read_csv("/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/cleaned_data.csv")

# Ensure all columns marked as numeric in data cleaning are numeric here (excluding categorical like 'existing_loan')
numeric_cols = [
    'age', 'monthly_salary', 'employment_years_of_exp', 'monthly_rent', 'family_size', 'dependents',
    'school_fees', 'college_fees', 'travel_expenses', 'groceries_util', 'other_month',
    'current_emi', 'credit_score', 'bank_balance', 'emergency_fund',
    'requested_amount', 'requested_tenure', 'max_monthly_emi'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 1. Derived Financial Ratios (synchronized with app logic)
print("\n=== Derived Financial Ratios ===")
def safe_col(colname, default=0):
    return df[colname] if colname in df.columns else default

# Use the same formulas as in the Streamlit app for consistency
if 'current_emi_amount' in df.columns:
    emi_col = 'current_emi_amount'
elif 'current_emi' in df.columns:
    emi_col = 'current_emi'
else:
    emi_col = None

if 'groceries_utilities' in df.columns:
    groceries_col = 'groceries_utilities'
elif 'groceries_util' in df.columns:
    groceries_col = 'groceries_util'
else:
    groceries_col = None

if 'other_monthly_expenses' in df.columns:
    other_exp_col = 'other_monthly_expenses'
elif 'other_month' in df.columns:
    other_exp_col = 'other_month'
else:
    other_exp_col = None

df['debt_to_income'] = (safe_col(emi_col, 0) + safe_col('requested_amount', 0) / (safe_col('requested_tenure', 1))) / (safe_col('monthly_salary', 1))
df['expense_to_income'] = (
    safe_col('monthly_rent', 0) + safe_col(groceries_col, 0) + safe_col(other_exp_col, 0) + safe_col('school_fees', 0) + safe_col('college_fees', 0) + safe_col('travel_expenses', 0)
) / (safe_col('monthly_salary', 1))
df['affordability_ratio'] = safe_col('max_monthly_emi', 0) / (safe_col('monthly_salary', 1))
df['requested_amount_to_income_ratio'] = safe_col('requested_amount', 0) / (safe_col('monthly_salary', 1))

# 2. Risk Scoring Features (synchronized with app logic)
print("\n=== Risk Scoring Features ===")
df['credit_history_good'] = (safe_col('credit_score', 0) >= 700).astype(int)
if 'employment_years_of_exp' in df.columns:
    df['employment_stable'] = (df['employment_years_of_exp'] >= 2).astype(int)
elif 'years_of_employment' in df.columns:
    df['employment_stable'] = (df['years_of_employment'] >= 2).astype(int)
else:
    df['employment_stable'] = 0

# 3. Categorical Encoding (do not drop first, to match app one-hot logic)
print("\n=== Categorical Encoding ===")
categorical_cols = [col for col in ['gender', 'marital_status', 'education', 'employment_type', 'company_type', 'house_type', 'emi_scenario'] if col in df.columns]

# Custom one-hot encoding for gender (only Male/Female columns)
df_encoded = df.copy()
if 'gender' in categorical_cols:
    df_encoded['gender_Male'] = (df['gender'] == 'Male').astype(int)
    df_encoded['gender_Female'] = (df['gender'] == 'Female').astype(int)
    categorical_cols = [col for col in categorical_cols if col != 'gender']
# One-hot encode remaining categorical columns
if categorical_cols:
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=False)
# Restore emi_eligibility column for modeling
if 'emi_eligibility' in df.columns:
    df_encoded['emi_eligibility'] = df['emi_eligibility']

# 4. Numerical Feature Scaling (optional, but keep for consistency)

print("\n=== Numerical Feature Scaling (after interaction features) ===")

# 5. Interaction Features (synchronized with app logic)
print("\n=== Interaction Features ===")
if 'monthly_salary' in df_encoded.columns and 'credit_score' in df_encoded.columns:
    df_encoded['income_x_credit'] = df_encoded['monthly_salary'] * df_encoded['credit_score']
if 'max_monthly_emi' in df_encoded.columns and 'requested_tenure' in df_encoded.columns:
    df_encoded['emi_x_tenure'] = df_encoded['max_monthly_emi'] * df_encoded['requested_tenure']
elif 'requested_amount' in df_encoded.columns and 'requested_tenure' in df_encoded.columns:
    df_encoded['emi_x_tenure'] = df_encoded['requested_amount'] * df_encoded['requested_tenure']

# Always create emi_x_tenure_requested as requested_amount * requested_tenure
if 'requested_amount' in df_encoded.columns and 'requested_tenure' in df_encoded.columns:
    df_encoded['emi_x_tenure_requested'] = df_encoded['requested_amount'] * df_encoded['requested_tenure']

# Now define numeric columns including interaction features
features_used_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/features_used.txt"
if os.path.exists(features_used_path):
    with open(features_used_path, "r") as f:
        features_used = [line.strip() for line in f.readlines() if line.strip()]
else:
    features_used = list(df_encoded.columns)
numeric_candidates = [
    'age', 'monthly_salary', 'years_of_employment', 'employment_years_of_exp', 'monthly_rent', 'family_size', 'dependents',
    'school_fees', 'college_fees', 'travel_expenses', 'groceries_util', 'groceries_utilities', 'other_month', 'other_monthly_expenses',
    'current_emi', 'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
    'requested_amount', 'requested_tenure', 'max_monthly_emi',
    'debt_to_income', 'expense_to_income', 'affordability_ratio',
    'income_x_credit', 'emi_x_tenure',
     'emi_x_tenure_requested',
    'requested_amount_to_income_ratio'
]
numeric_cols = [col for col in numeric_candidates if col in features_used and col in df_encoded.columns]
for col in numeric_cols:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Fit scaler on all numeric_cols at once (to preserve feature order)

# 5. Interaction Features (synchronized with app logic)
print("\n=== Interaction Features ===")
if 'monthly_salary' in df_encoded.columns and 'credit_score' in df_encoded.columns:
    df_encoded['income_x_credit'] = df_encoded['monthly_salary'] * df_encoded['credit_score']
if 'max_monthly_emi' in df_encoded.columns and 'requested_tenure' in df_encoded.columns:
    df_encoded['emi_x_tenure'] = df_encoded['max_monthly_emi'] * df_encoded['requested_tenure']
elif 'requested_amount' in df_encoded.columns and 'requested_tenure' in df_encoded.columns:
    df_encoded['emi_x_tenure'] = df_encoded['requested_amount'] * df_encoded['requested_tenure']

# Now define numeric columns including interaction features
features_used_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/features_used.txt"
if os.path.exists(features_used_path):
    with open(features_used_path, "r") as f:
        features_used = [line.strip() for line in f.readlines() if line.strip()]
else:
    features_used = list(df_encoded.columns)
numeric_candidates = [
    'age', 'monthly_salary', 'years_of_employment', 'employment_years_of_exp', 'monthly_rent', 'family_size', 'dependents',
    'school_fees', 'college_fees', 'travel_expenses', 'groceries_util', 'groceries_utilities', 'other_month', 'other_monthly_expenses',
    'current_emi', 'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
    'requested_amount', 'requested_tenure', 'max_monthly_emi',
    'debt_to_income', 'expense_to_income', 'affordability_ratio',
    'income_x_credit', 'emi_x_tenure',
     'emi_x_tenure_requested',
    'requested_amount_to_income_ratio'
]
numeric_cols = [col for col in numeric_candidates if col in features_used and col in df_encoded.columns]
for col in numeric_cols:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Fit scaler on all numeric_cols at once (to preserve feature order)


from sklearn.preprocessing import StandardScaler
import joblib

# Single scaler for all numeric_cols
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
numeric_cols_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/scaler_numeric_cols.txt"
with open(numeric_cols_path, "w") as f:
    for col in numeric_cols:
        f.write(col + "\n")
scaler_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/standard_scaler.joblib"
joblib.dump(scaler, scaler_path)
print(f"\n✓ Numeric column order for scaler saved to: {numeric_cols_path}")
print(f"✓ StandardScaler saved to: {scaler_path} with {len(numeric_cols)} columns")
print("Scaler mean_ (training):", list(scaler.mean_))
print("Scaler scale_ (training):", list(scaler.scale_))
print(f"✓ scaler_numeric_cols.txt updated with {len(numeric_cols)} columns including interaction features")

# Ensure all features in features_used.txt are present in df_encoded (add missing as 0)

features_used_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/features_used.txt"
if os.path.exists(features_used_path):
    with open(features_used_path, "r") as f:
        required_features = [line.strip() for line in f.readlines() if line.strip()]
    for col in required_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    # Remove any extra columns not in features_used.txt
    df_encoded = df_encoded[[col for col in required_features if col in df_encoded.columns]]
    print(f"✓ Feature columns aligned to features_used.txt ({len(df_encoded.columns)} features)")
else:
    print("Warning: features_used.txt not found. Saving all columns.")

# Always append emi_eligibility as the last column if present in df
if 'emi_eligibility' in df.columns:
    df_encoded['emi_eligibility'] = df['emi_eligibility']

# Ensure features_used.txt always includes emi_eligibility as last feature if present
features_used_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/features_used.txt"

# Only keep gender_Male and gender_Female in features_used.txt

feature_cols = [col for col in df_encoded.columns if not (col.startswith('gender_') and col not in ['gender_Male', 'gender_Female'])]
# Always add emi_x_tenure_requested to features if present
if 'emi_x_tenure_requested' in df_encoded.columns and 'emi_x_tenure_requested' not in feature_cols:
    if 'emi_x_tenure' in feature_cols:
        idx = feature_cols.index('emi_x_tenure') + 1
        feature_cols.insert(idx, 'emi_x_tenure_requested')
    elif 'emi_eligibility' in feature_cols:
        idx = feature_cols.index('emi_eligibility')
        feature_cols.insert(idx, 'emi_x_tenure_requested')
    else:
        feature_cols.append('emi_x_tenure_requested')
# Always add requested_amount_to_income_ratio to features if present
if 'requested_amount_to_income_ratio' in df_encoded.columns and 'requested_amount_to_income_ratio' not in feature_cols:
    # Insert after affordability_ratio if present, else after expense_to_income, else at end before emi_eligibility
    if 'affordability_ratio' in feature_cols:
        idx = feature_cols.index('affordability_ratio') + 1
        feature_cols.insert(idx, 'requested_amount_to_income_ratio')
    elif 'expense_to_income' in feature_cols:
        idx = feature_cols.index('expense_to_income') + 1
        feature_cols.insert(idx, 'requested_amount_to_income_ratio')
    elif 'emi_eligibility' in feature_cols:
        idx = feature_cols.index('emi_eligibility')
        feature_cols.insert(idx, 'requested_amount_to_income_ratio')
    else:
        feature_cols.append('requested_amount_to_income_ratio')
if 'emi_eligibility' in feature_cols and (not os.path.exists(features_used_path) or feature_cols[-1] != 'emi_eligibility'):
    with open(features_used_path, "w") as f:
        for col in feature_cols:
            f.write(col + "\n")
    print(f"✓ features_used.txt updated with only gender_Male and gender_Female as gender columns and requested_amount_to_income_ratio ({len(feature_cols)} features)")

output_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/feature_engineered_data.csv"

# Overwrite features_used.txt with only gender_Male and gender_Female as gender columns (emi_eligibility last if present)
import uuid
df_encoded = df_encoded.loc[:,~df_encoded.columns.duplicated()]
feature_cols = [col for col in df_encoded.columns if not (col.startswith('gender_') and col not in ['gender_Male', 'gender_Female'])]
feature_cols = [col for col in feature_cols if col != 'gender']
# Always ensure emi_eligibility is last if present
if 'emi_eligibility' in feature_cols:
    feature_cols = [col for col in feature_cols if col != 'emi_eligibility'] + ['emi_eligibility']
df_encoded = df_encoded[feature_cols]

# Ensure existing_loans is boolean if present
if 'existing_loans' in df_encoded.columns:

    # Convert safely: treat NaN as False, nonzero as True

    # Map string values to boolean, then convert numerics, treat NaN as False
    def to_bool_existing_loans(x):
        if pd.isnull(x):
            return False
        if isinstance(x, str):
            x_lower = x.strip().lower()
            if x_lower in ['yes', 'y', 'true', '1']:
                return True
            if x_lower in ['no', 'n', 'false', '0']:
                return False
            # fallback: try numeric
            try:
                return bool(int(float(x)))
            except Exception:
                return False
        try:
            return bool(int(float(x)))
        except Exception:
            return False
    df_encoded['existing_loans'] = df_encoded['existing_loans'].apply(to_bool_existing_loans)
features_used_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/features_used.txt"
with open(features_used_path, "w") as f:
    for col in feature_cols:
        f.write(col + "\n")
print(f"✓ features_used.txt overwritten with only gender_Male and gender_Female as gender columns ({len(feature_cols)} columns, emi_eligibility included if present)")
# Overwrite scaler_numeric_cols.txt with only those numeric columns present in feature_cols and unique
scaler_numeric_cols_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/scaler_numeric_cols.txt"
numeric_cols_final = [col for col in numeric_cols if col in feature_cols]
numeric_cols_final = list(dict.fromkeys(numeric_cols_final))
with open(scaler_numeric_cols_path, "w") as f:
    for col in numeric_cols_final:
        f.write(col + "\n")
print(f"✓ scaler_numeric_cols.txt overwritten with {len(numeric_cols_final)} columns")
# Save final DataFrame

# Also save features_used.txt for classification and regression
features_used_classification = [col for col in feature_cols if col not in ['emi_x_tenure_requested', 'requested_amount_to_income_ratio']]
features_used_regression = [col for col in feature_cols if col not in ['affordability_ratio', 'emi_x_tenure', 'max_monthly_emi']]
features_used_classification_path = features_used_path.replace('features_used.txt', 'features_used_classification.txt')
features_used_regression_path = features_used_path.replace('features_used.txt', 'features_used_regression.txt')
with open(features_used_classification_path, "w") as f:
    for col in features_used_classification:
        f.write(col + "\n")
print(f"✓ features_used_classification.txt saved with {len(features_used_classification)} features")
with open(features_used_regression_path, "w") as f:
    for col in features_used_regression:
        f.write(col + "\n")
print(f"✓ features_used_regression.txt saved with {len(features_used_regression)} features")

# Save classification features (omit emi_x_tenure_requested and requested_amount_to_income_ratio)
classification_cols = [col for col in feature_cols if col not in ['emi_x_tenure_requested', 'requested_amount_to_income_ratio']]
classification_path = output_path.replace('feature_engineered_data.csv', 'feature_engineered_data_classification.csv')
df_encoded[classification_cols].to_csv(classification_path, index=False)
print(f"\n✓ Classification feature-engineered data saved to: {classification_path}")

# Save regression features (omit affordability_ratio and emi_x_tenure)
regression_cols = [col for col in feature_cols if col not in ['affordability_ratio', 'emi_x_tenure', 'max_monthly_emi']]
regression_path = output_path.replace('feature_engineered_data.csv', 'feature_engineered_data_regression.csv')
df_encoded[regression_cols].to_csv(regression_path, index=False)
print(f"✓ Regression feature-engineered data saved to: {regression_path}")

# Save the default (all features) as before
df_encoded.to_csv(output_path, index=False)
print(f"\n✓ Feature-engineered data saved to: {output_path}")
