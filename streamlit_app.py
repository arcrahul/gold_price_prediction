import streamlit as st

import pandas as pd

st.title("Gold Price Prediction")
ticker = 'GC=F'  # Gold Futures
data = yf.download(ticker, period='1y')

st.line_chart(data['Close'])
