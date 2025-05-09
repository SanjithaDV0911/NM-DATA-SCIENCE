# NM-DATA-SCIENCE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset (downloaded from Kaggle)
df = pd.read_csv("train.csv")
df.head()

# Drop columns with too many missing values
df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

# Fill remaining missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert categorical features using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Define input and output
X = df.drop(['SalePrice', 'Id'], axis=1)
y = df['SalePrice']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 Score:", r2_score(y_test, y_pred))
    return y_pred

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Linear Regression Results:")
evaluate_model(lr, X_test, y_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest Results:")
evaluate_model(rf, X_test, y_test)

# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
print("XGBoost Results:")
evaluate_model(xgb, X_test, y_test)

# Plot top 20 important features from XGBoost
importances = xgb.feature_importances_
indices = np.argsort(importances)[-20:]
features = X.columns[indices]

plt.figure(figsize=(10, 6))
plt.title('Top 20 Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), features)
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()

import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("xgboost_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏡 House Price Predictor")

# Sample input fields
OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 5)
GrLivArea = st.number_input("Above Grade Living Area (sq ft)", 500, 5000, 1500)
GarageCars = st.selectbox("Garage Capacity", [0, 1, 2, 3, 4])
TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
FullBath = st.selectbox("Number of Full Bathrooms", [0, 1, 2, 3])

# Feature array
input_data = np.array([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath]])
input_scaled = scaler.transform(input_data)

if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    st.success(f"Estimated Sale Price: ₹{int(prediction[0]):,}")

import pickle
pickle.dump(xgb, open("xgboost_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
