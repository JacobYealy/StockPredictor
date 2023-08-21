import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json

# Fetch stock data
stock_data = yf.download("TSLA", start="2020-01-01", end="2021-01-01")
stock_data = stock_data['Close'].values
stock_data = stock_data.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
stock_data_normalized = scaler.fit_transform(stock_data)

# Prepare the data for LSTM
X, y = [], []
for i in range(50, len(stock_data_normalized)):
    X.append(stock_data_normalized[i-50:i, 0])
    y.append(stock_data_normalized[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=50, batch_size=32)

# Make predictions
test_data = yf.download("TSLA", start="2021-01-01", end="2021-02-01")['Close'].values
test_data = test_data.reshape(-1, 1)
scaled_test_data = scaler.transform(test_data)

X_test, y_test = [], []
for i in range(50, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-50:i, 0])
X_test = np.array(X_test)
print(X_test.shape) #REMOVE AFTER TEST
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(np.reshape(predicted_prices, (predicted_prices.shape[0], 1)))

# Generate data for plotting
plot_data = {
    'x': list(range(len(predicted_prices))),
    'y': predicted_prices.tolist(),
    # any other relevant data
}

plot_data_json = json.dumps(plot_data)
