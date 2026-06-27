import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Stock Price Prediction App")

stock_name = st.text_input("Enter Stock Symbol", "AAPL")
data = yf.download(stock_name, start="2015-01-01", end="2025-01-01")

st.subheader("Stock Data")
st.write(data.tail())
st.subheader("Closing Price Chart")
st.line_chart(data["Close"])

data["Next_Close"] = data["Close"].shift(-1)
data = data.dropna()

X = data[["Open", "High", "Low", "Close", "Volume"]]
y = data["Next_Close"]

split = int(len(data) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("Model Evaluation")
st.write("MAE:", mae)
st.write("RMSE:", rmse)

st.subheader("Actual vs Predicted Price")

result = pd.DataFrame({
"Actual": y_test.values,
"Predicted": y_pred
})

st.line_chart(result)

latest_data = X.tail(1)
next_day_prediction = model.predict(latest_data)

st.subheader("Next Day Prediction")
st.write("Predicted Next Day Closing Price:", next_day_prediction[0])
