"""
Step 2: Exploratory Data Analysis (EDA)
Analyze EMI eligibility patterns, correlations, demographics, and generate statistical summaries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data
cleaned_data_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/cleaned_data.csv"
df = pd.read_csv(cleaned_data_path)

# 1. EMI Eligibility Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='emi_eligibility', data=df)
plt.title('EMI Eligibility Distribution')
plt.xlabel('EMI Eligibility')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. EMI Eligibility by Scenario
plt.figure(figsize=(10,5))
sns.countplot(x='emi_scenario', hue='emi_eligibility', data=df)
plt.title('EMI Eligibility by Scenario')
plt.xlabel('EMI Scenario')
plt.ylabel('Count')
plt.legend(title='Eligibility')
plt.tight_layout()
plt.show()

# 3. Correlation Heatmap (Numerical Features)
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(14,10))
sns.heatmap(df[numeric_cols].corr(), annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (Numerical Features)')
plt.tight_layout()
plt.show()

# 4. Demographic Patterns: Gender, Marital Status, Education
fig, axes = plt.subplots(1, 3, figsize=(18,5))
sns.countplot(x='gender', hue='emi_eligibility', data=df, ax=axes[0])
axes[0].set_title('Gender vs EMI Eligibility')
sns.countplot(x='marital_status', hue='emi_eligibility', data=df, ax=axes[1])
axes[1].set_title('Marital Status vs EMI Eligibility')
sns.countplot(x='education', hue='emi_eligibility', data=df, ax=axes[2])
axes[2].set_title('Education vs EMI Eligibility')
plt.tight_layout()
plt.show()

# 5. Risk Factor Relationships: Credit Score, Employment, Existing Loans
plt.figure(figsize=(8,5))
sns.boxplot(x='emi_eligibility', y='credit_score', data=df)
plt.title('Credit Score by EMI Eligibility')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='emi_eligibility', y='years_of_employment', data=df)
plt.title('Years of Employment by EMI Eligibility')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='emi_eligibility', y='current_emi_amount', data=df)
plt.title('Current EMI Amount by EMI Eligibility')
plt.tight_layout()
plt.show()

# 6. Statistical Summaries
print("\n=== Statistical Summary (Numerical) ===")
print(df.describe())

print("\n=== Statistical Summary (Categorical) ===")
print(df.describe(include='object'))

# 7. Business Insights
print("\n=== Business Insights ===")
print(f"Eligibility Rate: {df['emi_eligibility'].value_counts(normalize=True).to_dict()}")
print(f"Average Credit Score (Eligible): {df[df['emi_eligibility']=='Eligible']['credit_score'].mean():.2f}")
print(f"Average Credit Score (Not Eligible): {df[df['emi_eligibility']=='Not_Eligible']['credit_score'].mean():.2f}")
print(f"Average Requested Amount (Eligible): {df[df['emi_eligibility']=='Eligible']['requested_amount'].mean():.2f}")
print(f"Average Requested Amount (Not Eligible): {df[df['emi_eligibility']=='Not_Eligible']['requested_amount'].mean():.2f}")
