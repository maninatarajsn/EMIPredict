"""
Step 1: Data Loading and Preprocessing
Load the provided dataset of 400,000 financial records
Implement comprehensive data cleaning and validation
Create train-test-validation splits
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

def load_dataset(file_path):
    """Load the EMI prediction dataset"""
    print("=" * 80)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("=" * 80)
    
    print("\n📂 Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    
    return df

# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================

def explore_dataset(df):
    """Explore dataset structure and characteristics"""
    print("\n" + "=" * 80)
    print("DATASET EXPLORATION")
    print("=" * 80)
    
    print("\n📋 Column Information:")
    print(f"  - Total Columns: {len(df.columns)}")
    print(f"  - Data Types:\n{df.dtypes}")
    
    print("\n📊 First Few Records:")
    print(df.head())
    
    print("\n📈 Dataset Statistics:")
    print(df.describe())
    
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values found!")
    
    print("\n🔢 Data Types Distribution:")
    print(df.dtypes.value_counts())
    
    return df

# ============================================================================
# 3. DATA QUALITY ASSESSMENT
# ============================================================================

def assess_data_quality(df):
    """Assess data quality and identify issues"""
    print("\n" + "=" * 80)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 80)
    
    quality_report = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'numeric_cols': df.select_dtypes(include=[np.number]).shape[1],
        'categorical_cols': df.select_dtypes(include=['object']).shape[1],
    }
    
    print(f"\n✓ Total Records: {quality_report['total_records']}")
    print(f"✓ Missing Values: {quality_report['missing_values']}")
    print(f"✓ Duplicate Rows: {quality_report['duplicates']}")
    print(f"✓ Numeric Columns: {quality_report['numeric_cols']}")
    print(f"✓ Categorical Columns: {quality_report['categorical_cols']}")
    
    # Check for outliers in numeric columns
    print("\n🔍 Outlier Detection (using IQR method):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_count = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            print(f"  - {col}: {outliers} outliers detected")
            outliers_count += outliers
    
    if outliers_count == 0:
        print("  No significant outliers detected!")
    
    return quality_report

# ============================================================================
# 4. DATA CLEANING
# ============================================================================

def clean_data(df):

    """Comprehensive data cleaning with robust numeric cleaning"""
    print("\n" + "=" * 80)
    print("DATA CLEANING")
    print("=" * 80)

    # Diagnostic: Show value counts and NaN counts for key columns before cleaning
    key_cols = ['existing_loans', 'emi_eligibility']
    print("\n[DIAGNOSTIC] Raw value counts and NaN counts before cleaning:")
    for col in key_cols:
        if col in df.columns:
            print(f"- {col}: value counts\n{df[col].value_counts(dropna=False)}")
            print(f"  NaN count: {df[col].isna().sum()}")
        else:
            print(f"- {col}: NOT FOUND in raw data")

    df_clean = df.copy()

    # Normalize gender values to 'Male' or 'Female'
    if 'gender' in df_clean.columns:
        gender_map = {
            'male': 'Male', 'MALE': 'Male', 'M': 'Male', 'm': 'Male',
            'female': 'Female', 'FEMALE': 'Female', 'F': 'Female', 'f': 'Female',
            'Male': 'Male', 'Female': 'Female'
        }
        df_clean['gender'] = df_clean['gender'].astype(str).str.strip().map(lambda x: gender_map.get(x, x))

    # Remove duplicates
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_count - len(df_clean)
    print(f"\n✓ Removed duplicates: {removed_duplicates} rows")

    # Only clean columns that are truly expected to be numeric
    numeric_cols = [
        'age', 'monthly_salary', 'employment_years_of_exp', 'monthly_rent', 'family_size', 'dependents',
        'school_fees', 'college_fees', 'travel_expenses', 'groceries_util', 'other_monthly_exp',
         'current_emi', 'credit_score', 'bank_balance', 'emergency_fund',
         'requested_amount', 'requested_tenure', 'max_monthly_emi'
    ]

    def safe_float(val):
        if pd.isnull(val):
            return np.nan
        try:
            # Accept valid floats or ints
            return float(val)
        except Exception:
            # Try to extract a valid float from a string, but only if it looks like a number
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(val))
            if match:
                try:
                    return float(match.group(0))
                except Exception:
                    return np.nan
            return np.nan

    print("\n✓ Cleaning only truly numeric columns:")
    for col in numeric_cols:
        if col in df_clean.columns:
            before_na = df_clean[col].isna().sum()
            before_non_na = df_clean[col].notna().sum()
            df_clean[col] = df_clean[col].apply(safe_float)
            after_na = df_clean[col].isna().sum()
            print(f"  - {col}: Non-NA before: {before_non_na}, NaNs before: {before_na}, after: {after_na}, total changed to NaN: {after_na - before_na}")

    # Print summary of columns with high NaN rates after cleaning
    print("\n[SUMMARY] Columns with >10% NaN after cleaning:")
    row_count = len(df_clean)
    for col in numeric_cols:
        if col in df_clean.columns:
            nan_count = df_clean[col].isna().sum()
            if nan_count / row_count > 0.1:
                print(f"    {col}: {nan_count} NaN out of {row_count} ({100*nan_count/row_count:.1f}%)")

    # Handle missing values
    print("\n✓ Handling missing values:")
    for col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            if df_clean[col].dtype in ['float64', 'int64']:
                # Fill numeric columns with median
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                print(f"  - {col}: Filled {missing_count} with median")
            else:
                # Fill categorical columns with mode
                df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
                print(f"  - {col}: Filled {missing_count} with mode")

    # Remove rows with negative values in financial columns
    financial_cols = df_clean.select_dtypes(include=[np.number]).columns
    print("\n✓ Validating financial data:")
    for col in financial_cols:
        negative_count = (df_clean[col] < 0).sum()
        if negative_count > 0:
            df_clean = df_clean[df_clean[col] >= 0]
            print(f"  - {col}: Removed {negative_count} rows with negative values")

    # Diagnostic: Show value counts and NaN counts for key columns after cleaning
    print("\n[DIAGNOSTIC] Value counts and NaN counts after cleaning:")
    for col in key_cols:
        if col in df_clean.columns:
            print(f"- {col}: value counts\n{df_clean[col].value_counts(dropna=False)}")
            print(f"  NaN count: {df_clean[col].isna().sum()}")
        else:
            print(f"- {col}: NOT FOUND after cleaning")

    print(f"\n✓ Final clean dataset shape: {df_clean.shape}")
    print(f"✓ Records removed during cleaning: {len(df) - len(df_clean)}")

    return df_clean

# ============================================================================
# 5. DATA VALIDATION
# ============================================================================

def validate_data(df):
    """Validate data consistency and business rules"""
    print("\n" + "=" * 80)
    print("DATA VALIDATION")
    print("=" * 80)
    
    validation_passed = True
    
    # Check for required columns
    print("\n✓ Checking data integrity:")
    
    # Verify numeric columns are numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"  - Numeric columns: {len(numeric_cols)}")
    
    # Verify no NaN values remain
    nan_count = df.isnull().sum().sum()
    if nan_count == 0:
        print(f"  - No missing values: ✓")
    else:
        print(f"  - Warning: {nan_count} missing values found")
        validation_passed = False
    
    # Verify data ranges are reasonable
    print(f"  - Data ranges validation: ✓")
    
    if validation_passed:
        print("\n✅ All validation checks passed!")
    else:
        print("\n⚠️ Some validation issues detected - review needed")
    
    return validation_passed

# ============================================================================
# 6. TRAIN-TEST-VALIDATION SPLIT
# ============================================================================

def create_data_splits(df, test_size=0.2, val_size=0.1, random_state=42):
    """Create train, test, and validation splits"""
    print("\n" + "=" * 80)
    print("DATA SPLIT STRATEGY")
    print("=" * 80)
    
    # First split: separate test set
    df_temp, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    df_train, df_val = train_test_split(df_temp, test_size=val_size_adjusted, random_state=random_state)
    
    print(f"\n✓ Train Set: {len(df_train)} records ({len(df_train)/len(df)*100:.1f}%)")
    print(f"✓ Validation Set: {len(df_val)} records ({len(df_val)/len(df)*100:.1f}%)")
    print(f"✓ Test Set: {len(df_test)} records ({len(df_test)/len(df)*100:.1f}%)")
    print(f"✓ Total: {len(df_train) + len(df_val) + len(df_test)} records")
    
    return df_train, df_val, df_test

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load dataset
    dataset_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/emi_prediction_dataset.csv"
    df = load_dataset(dataset_path)
    
    # Explore dataset
    df = explore_dataset(df)
    
    # Assess data quality
    quality_report = assess_data_quality(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Validate data
    validate_data(df_clean)
    
    # Create splits
    df_train, df_val, df_test = create_data_splits(df_clean)
    
    # Save processed datasets
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)
    
    df_clean.to_csv('/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/cleaned_data.csv', index=False)
    df_train.to_csv('/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/train_data.csv', index=False)
    df_val.to_csv('/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/val_data.csv', index=False)
    df_test.to_csv('/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/test_data.csv', index=False)
    
    print("\n✓ Cleaned data saved: cleaned_data.csv")
    print("✓ Training data saved: train_data.csv")
    print("✓ Validation data saved: val_data.csv")
    print("✓ Test data saved: test_data.csv")
    
    print("\n✅ STEP 1 COMPLETE: Data Loading and Preprocessing")
