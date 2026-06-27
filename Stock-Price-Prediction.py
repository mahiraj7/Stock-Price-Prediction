import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

stock_name = "AAPL"

data = yf.download(stock_name, start="2015-01-01", end="2025-01-01")

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

print("MAE:", mae)
print("RMSE:", rmse)

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Price")
plt.plot(y_pred, label="Predicted Price")
plt.title("Actual vs Predicted Stock Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.savefig("prediction_plot.png")
plt.close()

latest_data = X.tail(1)
next_day_prediction = model.predict(latest_data)
print("Predicted Next Day Closing Price:", next_day_prediction[0])
