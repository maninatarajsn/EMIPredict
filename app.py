import streamlit as st
# Ensure expected_features is loaded before company_type one-hot encoding
if 'expected_features' not in locals() and 'expected_features' not in globals():
	#feature_file = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/features_used_classification.txt"
	feature_file = "data/processed/features_used_classification.txt"
	from pathlib import Path
	root = Path(__file__).parent
	feature_path = root / "EMIPredict_AI" / "data" / "processed" / "features_used_classification.txt"
	if feature_path.exists():
		expected_features = feature_path.read_text().splitlines()
	else:
		expected_features = []
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
import streamlit as st

st.set_page_config(page_title="EMIPredict AI", layout="wide")

st.sidebar.title("EMIPredict AI Navigation")
page = st.sidebar.radio("Go to", [
	"Home",
	"EMI Eligibility Prediction",
	"Maximum EMI Prediction",
	#"Data Explorer",
	#"Model Performance",
	#"Admin/Data Management"
])

if page == "Home":
	st.title("EMIPredict AI")
	st.write("Welcome to EMIPredict AI! Use the sidebar to navigate.")

elif page == "EMI Eligibility Prediction":
	st.header("EMI Eligibility Prediction (Classification)")

	import joblib
	import mlflow.pyfunc
	import os
	import pandas as pd
	st.sidebar.title("EMIPredict AI Navigation")
	age = st.number_input('Age', min_value=18.0, max_value=75.0, value=30.0, step=1.0, format="%.2f")
	monthly_salary = st.number_input('Monthly Salary', min_value=0.0, value=50000.0, step=1000.0, format="%.2f")

	# Set only the one-hot columns for company_type that are present in features_used_classification.txt.
	monthly_rent = st.number_input('Current rent', min_value=0.0, value=3000.0, step=100.0, format="%.2f")
	years_of_employment = st.number_input('Years of employment', min_value=0.0, value=1.0, step=0.1, format="%.2f")
	family_size = st.number_input('Family Size', min_value=1.0, value=4.0, step=1.0, format="%.2f")
	dependents = st.number_input('Number of Dependents', min_value=0.0, value=2.0, step=1.0, format="%.2f")
	school_fees = st.number_input('School Fees', min_value=0.0, value=2000.0, step=100.0, format="%.2f")
	college_fees = st.number_input('College Fees', min_value=0.0, value=3000.0, step=100.0, format="%.2f")
	travel_expenses = st.number_input('Travel Expenses', min_value=0.0, value=2000.0, step=100.0, format="%.2f")
	groceries_utilities = st.number_input('Groceries & Utilities', min_value=0.0, value=8000.0, step=100.0, format="%.2f")
	other_monthly_expenses = st.number_input('Other Monthly Expenses', min_value=0.0, value=2000.0, step=100.0, format="%.2f")
	current_emi_amount = st.number_input('Current EMI Amount', min_value=0.0, value=0.0, step=100.0, format="%.2f")
	credit_score = st.number_input('Credit Score', min_value=300.0, max_value=900.0, value=700.0, step=1.0, format="%.2f")
	bank_balance = st.number_input('Bank Balance', min_value=0.0, value=50000.0, step=100.0, format="%.2f")
	emergency_fund = st.number_input('Emergency Fund', min_value=0.0, value=20000.0, step=100.0, format="%.2f")
	requested_amount = st.number_input('Requested Loan Amount', min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
	requested_tenure = st.number_input('Requested Tenure (months)', min_value=1.0, value=24.0, step=1.0, format="%.2f")

	# Categorical features

	gender = st.selectbox('Gender', [
		'Male', 'Female', 'male', 'female', 'MALE', 'FEMALE', 'M', 'F'])
	marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
	education = st.selectbox('Education', ['Graduate', 'Post Graduate', 'High School', 'Professional'])
	employment_type = st.selectbox('Employment Type', ['Private', 'Government', 'Self-employed'])
	company_type = st.selectbox('Company Type', ['MNC', 'Mid-size', 'Small', 'Startup', 'Large Indian'])
	house_type = st.selectbox('House Type', ['Own', 'Rented', 'Family'])
	emi_scenario = st.selectbox('EMI Scenario', [
		'Personal Loan EMI', 'Vehicle EMI', 'Home Appliances EMI', 'Education EMI', 'E-commerce Shopping EMI'])

	# Engineered features

	debt_to_income = float((current_emi_amount + requested_amount / max(requested_tenure,1))) / float(max(monthly_salary,1))
	expense_to_income = float(monthly_rent + groceries_utilities + other_monthly_expenses + school_fees + college_fees + travel_expenses) / float(max(monthly_salary,1))
	credit_history_good = 1 if credit_score >= 700 else 0
	employment_stable = 1 if years_of_employment >= 2 else 0
	income_x_credit = float(monthly_salary) * float(credit_score)
	max_monthly_emi = st.number_input('Max Monthly EMI Willing to Pay', min_value=0.0, value=15000.0, step=100.0, format="%.2f")
	affordability_ratio = float(max_monthly_emi) / float(max(monthly_salary,1))
	emi_x_tenure = float(max_monthly_emi) * float(requested_tenure) if max_monthly_emi and requested_tenure else (float(requested_amount) * float(requested_tenure) if requested_amount and requested_tenure else 0.0)
	emi_x_tenure_requested = float(requested_amount) * float(requested_tenure) if requested_amount and requested_tenure else 0.0
	requested_amount_to_income_ratio = (float(requested_amount) / float(monthly_salary)) if monthly_salary else 0.0

	# Calculate existing_loans as boolean: True if current_emi_amount > 0, else False
	existing_loans = current_emi_amount > 0

	# One-hot encoding for gender (only Male/Female columns)
	gender_Male = 1 if gender.strip().lower() == 'male' else 0
	gender_Female = 1 if gender.strip().lower() == 'female' else 0

	# Marital Status
	marital_status_Married = 1 if marital_status == 'Married' else 0
	marital_status_Single = 1 if marital_status == 'Single' else 0

	# Education
	education_Graduate = 1 if education == 'Graduate' else 0
	education_High_School = 1 if education == 'High School' else 0  # for 'education_High_School'
	education_High_School_space = 1 if education == 'High School' else 0  # for 'education_High School'
	education_Post_Graduate = 1 if education == 'Post Graduate' else 0  # for 'education_Post_Graduate'
	education_Post_Graduate_space = 1 if education == 'Post Graduate' else 0  # for 'education_Post Graduate'
	education_Professional = 1 if education == 'Professional' else 0

	# Employment Type
	employment_type_Government = 1 if employment_type == 'Government' else 0
	employment_type_Private = 1 if employment_type == 'Private' else 0
	employment_type_Self_employed = 1 if employment_type == 'Self-employed' else 0  # for 'employment_type_Self_employed'
	employment_type_Self_employed_dash = 1 if employment_type == 'Self-employed' else 0  # for 'employment_type_Self-employed'

	# Company Type

	# List of all possible company_type one-hot columns
	company_type_onehot_cols = [
		'company_type_Large Indian',
		'company_type_MNC',
		'company_type_Mid-size',
		'company_type_Mid_size',
		'company_type_Small',
		'company_type_Startup'
	]
	# Map UI value to all possible one-hot columns it could activate
	company_type_map = {
		'Large Indian': ['company_type_Large Indian'],
		'MNC': ['company_type_MNC'],
		'Mid-size': ['company_type_Mid-size', 'company_type_Mid_size'],
		'Small': ['company_type_Small'],
		'Startup': ['company_type_Startup']
	}
	# Only set those one-hot columns that are present in expected_features
	for col in company_type_onehot_cols:
		if col in expected_features:
			if col in company_type_map.get(company_type, []):
				locals()[col] = 1
			else:
				locals()[col] = 0

	# House Type
	house_type_Family = 0  # Not in UI, set to 0
	house_type_Own = 1 if house_type == 'Own' else 0
	house_type_Rented = 1 if house_type == 'Rented' else 0

	# EMI Scenario

	# Robust one-hot encoding for emi_scenario

	# Load expected_features before feature assignments if not already loaded
	if 'expected_features' not in locals():
		#feature_file = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/features_used_classification.txt"
		from pathlib import Path
		root = Path(__file__).parent
		feature_file = root / "EMIPredict_AI" / "data" / "processed" / "features_used_classification.txt"
		if feature_file.exists():
			expected_features = feature_file.read_text().splitlines()
		else:
			expected_features = []

	# List of all possible emi_scenario one-hot columns
	emi_scenario_onehot_cols = [
		'emi_scenario_E-commerce Shopping EMI',
		'emi_scenario_Education EMI',
		'emi_scenario_Education_EMI',
		'emi_scenario_Home Appliances EMI',
		'emi_scenario_Home_Appliances_EMI',
		'emi_scenario_Personal Loan EMI',
		'emi_scenario_Personal_Loan_EMI',
		'emi_scenario_Vehicle EMI',
		'emi_scenario_Vehicle_EMI'
	]
	# Map UI value to all possible one-hot columns it could activate
	emi_scenario_map = {
		'E-commerce Shopping EMI': ['emi_scenario_E-commerce Shopping EMI'],
		'Education EMI': ['emi_scenario_Education EMI', 'emi_scenario_Education_EMI'],
		'Home Appliances EMI': ['emi_scenario_Home Appliances EMI', 'emi_scenario_Home_Appliances_EMI'],
		'Personal Loan EMI': ['emi_scenario_Personal Loan EMI', 'emi_scenario_Personal_Loan_EMI'],
		'Vehicle EMI': ['emi_scenario_Vehicle EMI', 'emi_scenario_Vehicle_EMI']
	}
	# Only set those one-hot columns that are present in expected_features
	for col in emi_scenario_onehot_cols:
		if col in expected_features:
			# Set to 1 if this col is mapped from the selected emi_scenario, else 0
			if col in emi_scenario_map.get(emi_scenario, []):
				locals()[col] = 1
			else:
				locals()[col] = 0

	# Load expected_features before feature assignments, then build input_dict after all variables are set
	#feature_file = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/features_used_classification.txt"
	#scaler_cols_file = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/scaler_numeric_cols.txt"
	#scaler_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/standard_scaler.joblib"
	feature_file = root / "EMIPredict_AI" / "data" / "processed" / "features_used_classification.txt"
	scaler_cols_file = root / "EMIPredict_AI" / "data" / "processed" / "scaler_numeric_cols.txt"
	scaler_path = root / "EMIPredict_AI" / "data" / "processed" / "standard_scaler.joblib"

	if feature_file.exists():
		expected_features = feature_file.read_text().splitlines()
	else:
		expected_features = []

	# ...existing code for all feature assignments...

	# After all variables are set, build input_dict using globals() and locals()
	def get_var_value(var):
		# Prefer local, then global, else 0
		return locals().get(var, globals().get(var, 0))
	input_dict = {col: get_var_value(col) for col in expected_features}
	input_df = pd.DataFrame([input_dict])[expected_features]

	show_debug = st.checkbox("Show debug info (input DataFrames, scaler stats)", value=False)

	from pathlib import Path
	root = Path(__file__).parent
	scaler_cols_path = root / scaler_cols_file
	if scaler_cols_path.exists():
		scaler_numeric_cols = scaler_cols_path.read_text().splitlines()
	else:
		scaler_numeric_cols = []
	scaler_path_obj = root / scaler_path
	scaler = joblib.load(scaler_path_obj) if scaler_path_obj.exists() else None

	# Ensure required scaler columns are present, defaulting to 0 if missing
	for col in ['emi_x_tenure_requested', 'requested_amount_to_income_ratio']:
		if col not in input_df.columns:
			input_df[col] = 0.0
	scaler_numeric_cols = [col for col in scaler_numeric_cols if col in input_df.columns]
	st.dataframe(scaler_numeric_cols)

	if scaler is not None and scaler_numeric_cols:
		scaled_numeric = scaler.transform(input_df[scaler_numeric_cols])
		input_df_scaled = input_df.copy()
		input_df_scaled[scaler_numeric_cols] = scaled_numeric
	else:
		input_df_scaled = input_df.copy()

	if show_debug:
		st.subheader("Model Input DataFrame (pre-scaling)")
		st.dataframe(input_df)
		st.subheader("Model Input DataFrame (post-scaling)")
		st.dataframe(input_df_scaled)
		if scaler is None:
			st.warning("Scaler object is None. Scaling will not be applied.")
		st.write(f"scaler_numeric_cols: {scaler_numeric_cols}")
		if scaler_numeric_cols:
			st.subheader("Scaled Numeric Columns Only")
			st.dataframe(input_df_scaled[scaler_numeric_cols])
			st.write("Input values before scaling:")
			st.dataframe(input_df[scaler_numeric_cols])
			if scaler is not None:
				st.write(f"scaler.mean_: {getattr(scaler, 'mean_', None)}")
				st.write(f"scaler.scale_: {getattr(scaler, 'scale_', None)}")
				scaler_stats = pd.DataFrame({
					'mean_': getattr(scaler, 'mean_', [None]*len(scaler_numeric_cols)),
					'scale_': getattr(scaler, 'scale_', [None]*len(scaler_numeric_cols))
				}, index=scaler_numeric_cols)
				st.subheader("Scaler Mean and Scale Values")
				st.dataframe(scaler_stats)

	# Load MLflow model (BestEMIClassifier, alias production)
	model_path = root / "EMIPredict_AI" / "data" / "processed" / "emiclassifier.pkl"
	try:
		model = joblib.load(model_path)
		st.success("Loaded BestEMIClassifier from MLflow (production)")
	except Exception as e:
		model = None
		st.error(f"Could not load MLflow model: {e}")

	threshold = st.slider("Eligibility Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="Increase to make the model stricter")
	if st.button("Predict Eligibility"):
		if model is not None:
			# Remove any target column if present and align to features_used.txt (minus 'emi_eligibility')
			model_features = [col for col in expected_features if col != 'emi_eligibility']

			# Drop scaler-only columns before sending to model
			drop_for_model = ['emi_x_tenure_requested', 'requested_amount_to_income_ratio']
			model_input = input_df_scaled.reindex(columns=model_features, fill_value=0)
			for col in drop_for_model:
				if col in model_input.columns:
					model_input = model_input.drop(columns=[col])

			# Explicitly cast columns to match model schema
			int_cols = ["credit_history_good", "employment_stable", "gender_Male", "gender_Female"]
			bool_cols = [
				'existing_loans',
				'marital_status_Married', 'marital_status_Single',
				'education_Graduate', 'education_High School', 'education_High_School', 'education_Post Graduate', 'education_Post_Graduate', 'education_Professional',
				'employment_type_Government', 'employment_type_Private', 'employment_type_Self-employed', 'employment_type_Self_employed',
				'company_type_Large Indian', 'company_type_MNC', 'company_type_Mid-size', 'company_type_Mid_size', 'company_type_Small', 'company_type_Startup',
				'house_type_Family', 'house_type_Own', 'house_type_Rented',
				'emi_scenario_E-commerce Shopping EMI', 'emi_scenario_Education EMI', 'emi_scenario_Education_EMI', 'emi_scenario_Home Appliances EMI', 'emi_scenario_Home_Appliances_EMI',
				'emi_scenario_Personal Loan EMI', 'emi_scenario_Personal_Loan_EMI', 'emi_scenario_Vehicle EMI', 'emi_scenario_Vehicle_EMI'
			]
			for col in int_cols:
				if col in model_input.columns:
					model_input[col] = model_input[col].astype(int)
			for col in bool_cols:
				if col in model_input.columns:
					model_input[col] = model_input[col].astype(bool)
			# All others to float (if not already int or bool)
			for col in model_input.columns:
				if col not in int_cols and col not in bool_cols:
					model_input[col] = model_input[col].astype(float)
			try:
				proba = model.predict(model_input)[0]
				pred = int(proba >= threshold)
				st.write(f"Predicted probability of eligibility: {proba:.4f}")
				st.write(f"Raw model output (predict): {model.predict(model_input)}")
				st.success(f"Prediction: {'Eligible' if pred == 1 else 'Not Eligible'} (Threshold: {threshold})")
			except Exception as e:
				st.error(f"Prediction failed: {e}")
		else:
			st.warning("Model or scaler not found. Please check model files.")

elif page == "Maximum EMI Prediction":
	st.header("Maximum EMI Amount Prediction (Regression)")
	import joblib
	import mlflow.pyfunc
	import os
	import pandas as pd
	# Numeric inputs (same as eligibility, but exclude max_monthly_emi)
	age = st.number_input('Age', min_value=18.0, max_value=75.0, value=30.0, step=1.0, format="%.2f")
	monthly_salary = st.number_input('Monthly Salary', min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
	years_of_employment = st.number_input('Years of Employment', min_value=0.0, value=5.0, step=1.0, format="%.2f")
	monthly_rent = st.number_input('Monthly Rent', min_value=0.0, value=10000.0, step=500.0, format="%.2f")
	family_size = st.number_input('Family Size', min_value=1.0, value=4.0, step=1.0, format="%.2f")
	dependents = st.number_input('Number of Dependents', min_value=0.0, value=2.0, step=1.0, format="%.2f")
	school_fees = st.number_input('School Fees', min_value=0.0, value=2000.0, step=100.0, format="%.2f")
	college_fees = st.number_input('College Fees', min_value=0.0, value=3000.0, step=100.0, format="%.2f")
	travel_expenses = st.number_input('Travel Expenses', min_value=0.0, value=2000.0, step=100.0, format="%.2f")
	groceries_utilities = st.number_input('Groceries & Utilities', min_value=0.0, value=8000.0, step=100.0, format="%.2f")
	other_monthly_expenses = st.number_input('Other Monthly Expenses', min_value=0.0, value=2000.0, step=100.0, format="%.2f")
	current_emi_amount = st.number_input('Current EMI Amount', min_value=0.0, value=0.0, step=100.0, format="%.2f")
	credit_score = st.number_input('Credit Score', min_value=300.0, max_value=900.0, value=700.0, step=1.0, format="%.2f")
	bank_balance = st.number_input('Bank Balance', min_value=0.0, value=50000.0, step=100.0, format="%.2f")
	emergency_fund = st.number_input('Emergency Fund', min_value=0.0, value=20000.0, step=100.0, format="%.2f")
	requested_amount = st.number_input('Requested Loan Amount', min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
	requested_tenure = st.number_input('Requested Tenure (months)', min_value=1.0, value=24.0, step=1.0, format="%.2f")

	# Categorical features

	gender = st.selectbox('Gender', [
		'Male', 'Female', 'male', 'female', 'MALE', 'FEMALE', 'M', 'F'])
	marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
	education = st.selectbox('Education', ['Graduate', 'Post Graduate', 'High School', 'Professional'])
	employment_type = st.selectbox('Employment Type', ['Private', 'Government', 'Self-employed'])
	company_type = st.selectbox('Company Type', ['MNC', 'Mid-size', 'Small', 'Startup', 'Large Indian'])
	house_type = st.selectbox('House Type', ['Own', 'Rented', 'Family'])
	emi_scenario = st.selectbox('EMI Scenario', [
		'Personal Loan EMI', 'Vehicle EMI', 'Home Appliances EMI', 'Education EMI', 'E-commerce Shopping EMI'])

	# Engineered features (exclude max_monthly_emi, but keep others)
	debt_to_income = float((current_emi_amount + requested_amount / max(requested_tenure,1))) / float(max(monthly_salary,1))
	expense_to_income = float(monthly_rent + groceries_utilities + other_monthly_expenses + school_fees + college_fees + travel_expenses) / float(max(monthly_salary,1))
	affordability_ratio = float(requested_amount) / float(max(monthly_salary,1))
	credit_history_good = 1 if credit_score >= 700 else 0
	employment_stable = 1 if years_of_employment >= 2 else 0
	income_x_credit = float(monthly_salary) * float(credit_score)
	if requested_amount and requested_tenure:
		emi_x_tenure = float(requested_amount) * float(requested_tenure)
	else:
		emi_x_tenure = 0.0
	emi_x_tenure_requested = requested_amount * requested_tenure
	requested_amount_to_income_ratio = requested_amount / monthly_salary

	# Ensure expected_features is loaded before one-hot encoding
	if 'expected_features' not in locals() and 'expected_features' not in globals():
		#feature_file = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/features_used_regression.txt"
		from pathlib import Path
		root = Path(__file__).parent
		feature_file = root / "EMIPredict_AI" / "data" / "processed" / "features_used_regression.txt"
		if feature_file.exists():
			expected_features = feature_file.read_text().splitlines()
		else:
			expected_features = []

	# One-hot encoding for gender (only Male/Female columns)
	gender_Male = 1 if gender.strip().lower() == 'male' else 0
	gender_Female = 1 if gender.strip().lower() == 'female' else 0

	# Marital Status
	marital_status_onehot_cols = ['marital_status_Married', 'marital_status_Single']
	for col in marital_status_onehot_cols:
		if col in expected_features:
			locals()[col] = 1 if col == f'marital_status_{marital_status}' else 0

	# Education
	education_onehot_cols = [
		'education_Graduate', 'education_High_School', 'education_High_School_space',
		'education_Post_Graduate', 'education_Post_Graduate_space', 'education_Professional'
	]
	education_map = {
		'Graduate': ['education_Graduate'],
		'High School': ['education_High_School', 'education_High_School_space'],
		'Post Graduate': ['education_Post_Graduate', 'education_Post_Graduate_space'],
		'Professional': ['education_Professional']
	}
	for col in education_onehot_cols:
		if col in expected_features:
			locals()[col] = 1 if col in education_map.get(education, []) else 0

	# Employment Type
	employment_type_onehot_cols = [
		'employment_type_Government', 'employment_type_Private',
		'employment_type_Self_employed', 'employment_type_Self_employed_dash'
	]
	employment_type_map = {
		'Government': ['employment_type_Government'],
		'Private': ['employment_type_Private'],
		'Self-employed': ['employment_type_Self_employed', 'employment_type_Self_employed_dash']
	}
	for col in employment_type_onehot_cols:
		if col in expected_features:
			locals()[col] = 1 if col in employment_type_map.get(employment_type, []) else 0

	# Company Type (only set columns present in expected_features)
	company_type_onehot_cols = [
		'company_type_Large Indian', 'company_type_MNC', 'company_type_Mid-size',
		'company_type_Mid_size', 'company_type_Small', 'company_type_Startup'
	]
	company_type_map = {
		'Large Indian': ['company_type_Large Indian'],
		'MNC': ['company_type_MNC'],
		'Mid-size': ['company_type_Mid-size', 'company_type_Mid_size'],
		'Small': ['company_type_Small'],
		'Startup': ['company_type_Startup']
	}
	for col in company_type_onehot_cols:
		if col in expected_features:
			if col in company_type_map.get(company_type, []):
				locals()[col] = 1
			else:
				locals()[col] = 0

	# House Type
	house_type_onehot_cols = ['house_type_Family', 'house_type_Own', 'house_type_Rented']
	house_type_map = {
		'Own': ['house_type_Own'],
		'Rented': ['house_type_Rented']
	}
	for col in house_type_onehot_cols:
		if col in expected_features:
			locals()[col] = 1 if col in house_type_map.get(house_type, []) else 0
	if 'house_type_Family' in expected_features:
		locals()['house_type_Family'] = 0

	# EMI Scenario (only set columns present in expected_features)
	emi_scenario_onehot_cols = [
		'emi_scenario_E-commerce Shopping EMI',
		'emi_scenario_Education EMI', 'emi_scenario_Education_EMI',
		'emi_scenario_Home Appliances EMI', 'emi_scenario_Home_Appliances_EMI',
		'emi_scenario_Personal Loan EMI', 'emi_scenario_Personal_Loan_EMI',
		'emi_scenario_Vehicle EMI', 'emi_scenario_Vehicle_EMI'
	]
	emi_scenario_map = {
		'E-commerce Shopping EMI': ['emi_scenario_E-commerce Shopping EMI'],
		'Education EMI': ['emi_scenario_Education EMI', 'emi_scenario_Education_EMI'],
		'Home Appliances EMI': ['emi_scenario_Home Appliances EMI', 'emi_scenario_Home_Appliances_EMI'],
		'Personal Loan EMI': ['emi_scenario_Personal Loan EMI', 'emi_scenario_Personal_Loan_EMI'],
		'Vehicle EMI': ['emi_scenario_Vehicle EMI', 'emi_scenario_Vehicle_EMI']
	}
	for col in emi_scenario_onehot_cols:
		if col in expected_features:
			if col in emi_scenario_map.get(emi_scenario, []):
				locals()[col] = 1
			else:
				locals()[col] = 0

	# Load expected_features before feature assignments, then build input_dict after all variables are set
	#feature_file = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/features_used_regression.txt"
	#scaler_cols_file = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/scaler_numeric_cols.txt"
	#scaler_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/standard_scaler.joblib"
	feature_file = root / "EMIPredict_AI" / "data" / "processed" / "features_used_regression.txt"
	scaler_cols_file = root / "EMIPredict_AI" / "data" / "processed" / "scaler_numeric_cols.txt"
	scaler_path = root / "EMIPredict_AI" / "data" / "processed" / "standard_scaler.joblib"

	if feature_file.exists():
		expected_features = feature_file.read_text().splitlines()
	else:
		expected_features = []

	# ...existing code for all feature assignments...

	# After all variables are set, build input_dict using globals() and locals()
	def get_var_value(var):
		# Prefer local, then global, else 0
		return locals().get(var, globals().get(var, 0))
	# Ensure max_monthly_emi is present in expected_features for regressor, but set to 0 in input_dict
	#if 'max_monthly_emi' not in expected_features:
	#	expected_features.append('max_monthly_emi')
	input_dict = {col: (0 if col == 'max_monthly_emi' else get_var_value(col)) for col in expected_features}
	input_df = pd.DataFrame([input_dict])[expected_features]

	show_debug = st.checkbox("Show debug info (input DataFrames, scaler stats)", value=False)

	scaler_cols_path = root / scaler_cols_file
	if scaler_cols_path.exists():
		scaler_numeric_cols = scaler_cols_path.read_text().splitlines()
	else:
		scaler_numeric_cols = []
	scaler_path_obj = root / scaler_path
	scaler = joblib.load(scaler_path_obj) if scaler_path_obj.exists() else None
	# Ensure affordability_ratio and emi_x_tenure columns are present for scaler, default to 0.0 if missing
	for col in ['affordability_ratio', 'emi_x_tenure']:
		if col not in input_df.columns:
			input_df[col] = 0.0
	# Omit affordability_ratio and emi_x_tenure from scaler columns for scaling, but keep them in input_df
	scaler_numeric_cols = [col for col in scaler_numeric_cols if col in input_df.columns and col not in ['affordability_ratio', 'emi_x_tenure']]

	if scaler is not None and scaler_numeric_cols:
		# Pass all scaler columns (including those omitted from scaling) to scaler.transform
		scaler_all_cols = scaler.get_feature_names_out() if hasattr(scaler, 'get_feature_names_out') else scaler.feature_names_in_
		# Ensure all scaler columns are present in input_df
		for col in scaler_all_cols:
			if col not in input_df.columns:
				input_df[col] = 0.0
		scaled_numeric = scaler.transform(input_df[scaler_all_cols])
		input_df_scaled = input_df.copy()
		input_df_scaled[list(scaler_all_cols)] = scaled_numeric
	else:
		input_df_scaled = input_df.copy()

	if show_debug:
		st.subheader("Model Input DataFrame (pre-scalingregr)")
		st.dataframe(input_df)
		st.subheader("Model Input DataFrame (post-scalingregr)")
		st.dataframe(input_df_scaled)
		if scaler is not None:
			# Use scaler's feature names for index to match mean_ and scale_ arrays
			scaler_feature_names = list(getattr(scaler, 'get_feature_names_out', lambda: getattr(scaler, 'feature_names_in_', []))())
			st.subheader("Scaler Feature Names")
			st.write(scaler_feature_names)
			scaler_stats = pd.DataFrame({
				'mean_': getattr(scaler, 'mean_', [None]*len(scaler_feature_names)),
				'scale_': getattr(scaler, 'scale_', [None]*len(scaler_feature_names))
			}, index=scaler_feature_names)
			st.subheader("Scaler Mean and Scale Values")
			st.dataframe(scaler_stats)

	# Load MLflow model (BestEMIRegressor, alias production)

	model_path = root / "EMIPredict_AI" / "data" / "processed" / "emiregressor.pkl"
	try:
		model = joblib.load(model_path)
		st.success("Loaded BestEMIRegressor from MLflow (production)")
	except Exception as e:
		model = None
		st.error(f"Could not load MLflow model: {e}")

	predict_btn = st.button("Predict Maximum EMI")
	if predict_btn:
		if model is not None:
			# Ensure max_monthly_emi and affordability_ratio are set to 0 in model input, and emi_eligibility is not passed
			model_features = [col for col in expected_features if col != 'emi_eligibility']
			model_input = input_df_scaled.reindex(columns=model_features, fill_value=0)

			# Hardcode requested_amount_to_income_ratio and emi_x_tenure_requested before passing to model

			#if 'max_monthly_emi' in model_input.columns:
			#	model_input['max_monthly_emi'] = 0
			#if 'affordability_ratio' in model_input.columns:
			#	model_input['affordability_ratio'] = 0
			st.header("Model Input DataFrame (for Mani)")
			st.dataframe(model_input)
			int_cols = ["credit_history_good", "employment_stable", "gender_Male", "gender_Female"]
			bool_cols = [
				'existing_loans',
				'marital_status_Married', 'marital_status_Single',
				'education_Graduate', 'education_High School', 'education_High_School', 'education_Post Graduate', 'education_Post_Graduate', 'education_Professional',
				'employment_type_Government', 'employment_type_Private', 'employment_type_Self-employed', 'employment_type_Self_employed',
				'company_type_Large Indian', 'company_type_MNC', 'company_type_Mid-size', 'company_type_Mid_size', 'company_type_Small', 'company_type_Startup',
				'house_type_Family', 'house_type_Own', 'house_type_Rented',
				'emi_scenario_E-commerce Shopping EMI', 'emi_scenario_Education EMI', 'emi_scenario_Education_EMI', 'emi_scenario_Home Appliances EMI', 'emi_scenario_Home_Appliances_EMI',
				'emi_scenario_Personal Loan EMI', 'emi_scenario_Personal_Loan_EMI', 'emi_scenario_Vehicle EMI', 'emi_scenario_Vehicle_EMI'
			]
			for col in int_cols:
				if col in model_input.columns:
					model_input[col] = model_input[col].astype(int)
			for col in bool_cols:
				if col in model_input.columns:
					model_input[col] = model_input[col].astype(bool)
			for col in model_input.columns:
				if col not in int_cols and col not in bool_cols:
					model_input[col] = model_input[col].astype(float)

			try:
				import numpy as np

				# Load the target scaler (StandardScaler for max_monthly_emi)
				#scaler_target_path = "/Users/m0s0pdp/Library/CloudStorage/OneDrive-WalmartInc/Documents/GUVI/EMIPredict_AI/data/processed/standard_scaler.joblib"
				from pathlib import Path
				root = Path(__file__).parent
				scaler_target_path = root / "EMIPredict_AI" / "data" / "processed" / "standard_scaler.joblib"
				import joblib
				scaler_target = None
				if scaler_target_path.exists():
					scaler_target = joblib.load(scaler_target_path)
				y_pred_scaled = float(model.predict(model_input)[0])
				if scaler_target is not None:
					# Unscale the prediction
					arr = np.zeros((1, scaler_target.n_features_in_))
					arr[0, 17] = y_pred_scaled
					arr_unscaled = scaler_target.inverse_transform(arr)
					y_pred_unscaled = arr_unscaled[0, 17]
					st.success(f"Predicted Maximum EMI: ₹{y_pred_unscaled:,.2f}")
				else:
					st.success(f"Predicted Maximum EMI (scaled): {y_pred_scaled:.4f}")
			except Exception as e:
				st.error(f"Prediction failed: {e}")
		else:
			st.warning("Model or scaler not found. Please check model files.")
