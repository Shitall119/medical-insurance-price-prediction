# 🏥 Medical Insurance Price Prediction

## Problem Statement :
Medical insurance costs vary significantly based on factors such as age, gender, BMI, number of children, smoking status, and region. The goal of this project is to provide an estimated insurance cost based on individual health and demographic inputs using Machine Learning models trained on historical data.

This can help users:
- Understand potential insurance costs in advance.
- Make informed decisions when comparing insurance policies.
- Focus on health-related aspects of insurance rather than pricing confusion.

---
## Dataset :
- **Source:** Kaggle – https://www.kaggle.com/datasets/rahulvyasm/medical-insurance-cost-prediction
- **File:** medical_insurance.csv
- The dataset includes the following features:

  age, sex, bmi, children, smoker, region, charges

---

## Approach :
Applying machine learing tasks like Data Exploration, Data Cleaning, Feature Engineering, Model Building and model testing to build a solution that should able to predict the price of the health insurance.
Here, a full machine learning pipeline including:

**1. Data Exploration** : Exploring the dataset using pandas, numpy, matplotlib, plotly and seaborn.structure and distributions in the dataset.
- Performed correlation analysis and visualizations to understand relationships between variables.
- Plotted different graphs to get more insights about dependent and independent variables/features.

**2. EDA :Data Cleaning & Feature Engineering**:
- Categorical Encoding: Used label encoding and one-hot encoding for non-numeric variables (sex, smoker, region).
- Scaling: Applied standard scaling to numerical features like bmi, age, etc.

**3. Model Building** : Tested the following machine learning regression models::
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. XGBoost Regressor

**4. Model Selection  & Evaluation** :
- Evaluated models based on **Root Mean Squared Error (RMSE)** and **R-squared (R²)** scores.
-Selected the model with the best performance metrics.

**5. Pickle File** : Serialized the final model using pickle library.

**6. Webpage** : A simple and interactive web application was developed using **Streamlit**:
- Takes user input through sliders and dropdowns.
- Displays the predicted insurance cost instantly.
- Designed for non-technical users to interact with the model easily..
  
---
## Web Inerface :
<img width="1919" height="857" alt="Screenshot 2025-07-29 144449" src="https://github.com/user-attachments/assets/4d7a30b8-f6ba-479f-bda9-e0b418f223fd" />
<img width="1919" height="862" alt="Screenshot 2025-07-29 144706" src="https://github.com/user-attachments/assets/4bb6d7d8-9c1d-48e1-b2cd-aee442adc8a5" />

---
## Libraries used :
- pandas – Data manipulation
- numpy – Numerical operations
- matplotlib & seaborn – Data visualization
- scikit-learn – ML models and preprocessing
- xgboost – Advanced gradient boosting
- streamlit – Web app deployment
- pickle – Model serialization

---
## ▶️ How to Run the Project
**1.Clone the Repository**
git clone https://github.com/Shitall119/medical-insurance-price-prediction.git
cd medical-insurance-price-prediction

**2.Create a Virtual Environment (optional)**

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

**3.Install Dependencies**

pip install -r requirements.txt

**4.Run the Web App**
streamlit run app.py

The app will open in your browser at http://localhost:8501.

---
## 📈 Results:
Best model: XGBoost Regressor (assumed, update if needed)

- R² Score: 0.9234 
- RMSE: 3427.93

  ---

