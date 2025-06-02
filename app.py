import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Load and preprocess data
df = pd.read_csv('medical_insurance.csv')

# Map categorical to numeric with updated smoker encoding: yes=1, no=0
df.replace({'sex': {'male': 0, 'female': 1},
            'smoker': {'yes': 1, 'no': 0},
            'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)

# Select only original features
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
}

# Train and evaluate models
model_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    model_scores[name] = score

# Select best model
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]

# Streamlit UI
st.title("🩺 Medical Insurance Charge Predictor")
st.markdown(f"Best performing model: **{best_model_name}** with R² = **{model_scores[best_model_name]:.2f}**")

with st.expander("📊 Model Comparison"):
    for name, score in model_scores.items():
        st.write(f"{name}: R² = {score:.4f}")

st.markdown("## 🧾 Enter person's data")

# User inputs
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sex_str = st.selectbox("Sex", options=["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker_str = st.selectbox("Smoker", options=["Yes", "No"])
region_str = st.selectbox("Region", options=["Southeast", "Southwest", "Northwest", "Northeast"])

# Map user inputs to numeric with updated smoker encoding: Yes=1, No=0
sex = 0 if sex_str == "Male" else 1
smoker = 1 if smoker_str == "Yes" else 0
region_map = {"Southeast": 0, "Southwest": 1, "Northwest": 2, "Northeast": 3}
region = region_map[region_str]

# Prepare input array
input_data = np.array([[age, sex, bmi, children, smoker, region]])

# Predict button
if st.button("Predict Insurance Charges"):
    prediction = best_model.predict(input_data)
    st.success(f"Estimated Medical Insurance Charges: **{prediction[0]:,.2f}**")
