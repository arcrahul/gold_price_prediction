import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Title
st.title("Gold Price Prediction App")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("gld_price_data.csv")

gold_data = load_data()

# Display data
if st.checkbox("Show raw data"):
    st.write(gold_data.head())

# Data preparation
X = gold_data.drop(['Date','GLD'], axis=1)
Y = gold_data['GLD']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

# R² score
r2 = metrics.r2_score(Y_test, predictions)
st.write(f"R² Score: {r2:.4f}")

# Plot
st.subheader("Actual vs Predicted Prices")
fig, ax = plt.subplots()
ax.plot(Y_test.values, label='Actual', color='blue')
ax.plot(predictions, label='Predicted', color='green')
plt.legend()
st.pyplot(fig)

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
st.subheader("Feature Importances")
imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
st.bar_chart(imp_df.set_index('Feature'))
