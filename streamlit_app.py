import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

st.set_option('deprecation.showPyplotGlobalUse', False)

# Title
st.title("Gold Price Prediction using Random Forest")

# Upload CSV
uploaded_file = st.file_uploader("Upload your gold price CSV file", type=['csv'])

if uploaded_file is not None:
    gold_data = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(gold_data.head())

    st.subheader("Statistical Summary")
    st.write(gold_data.describe())

    # Drop non-numeric columns before correlation
    correlation = gold_data.select_dtypes(include='number').corr()

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(8,8))
    sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',
                annot=True, annot_kws={'size':8}, cmap='Blues')
    st.pyplot()

    st.subheader("GLD Price Distribution")
    sns.histplot(gold_data['GLD'], kde=True, color='green')
    st.pyplot()

    # Data Preprocessing
    X = gold_data.drop(['Date', 'GLD'], axis=1)
    Y = gold_data['GLD']

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Model training
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(X_train, Y_train)

    # Predictions
    test_data_prediction = regressor.predict(X_test)

    # Evaluation
    error_score = metrics.r2_score(Y_test, test_data_prediction)
    st.subheader(f"R² Score: {error_score:.4f}")

    # Visualization
    st.subheader("Actual vs Predicted GLD Prices")
    fig, ax = plt.subplots()
    ax.plot(list(Y_test), color='blue', label='Actual Value')
    ax.plot(test_data_prediction, color='green', label='Predicted Value')
    ax.set_xlabel("Index")
    ax.set_ylabel("GLD Price")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin.")
