from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import yfinance as yf
import json

# Parameters
columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
look_back = 50
epochs = 50
batch_size = 32

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


def generate_plot_data():
    # Download stock data
    stock_data = yf.download("TSLA", start="2018-01-01", end="2022-01-01")
    plot_data_list = []

    # Loop over each column to make predictions
    for column in columns:
        print(f"Predicting {column}")

        # Extract and reshape data
        data = stock_data[column].values.reshape(-1, 1)

        # Normalize the data
        data_normalized = scaler.fit_transform(data)

        # Prepare the data for LSTM
        X, y = [], []
        for i in range(look_back, len(data_normalized)):
            X.append(data_normalized[i - look_back:i, 0])
            y.append(data_normalized[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        # Compile and train the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=epochs, batch_size=batch_size)

        # Make predictions using test data
        test_data = yf.download("TSLA", start="2018-01-01", end="2022-01-01")[column].values.reshape(-1, 1)
        scaled_test_data = scaler.transform(test_data)

        X_test = []
        print("Shape of scaled_test_data:", scaled_test_data.shape)
        for i in range(look_back, len(scaled_test_data)):
            X_test.append(scaled_test_data[i - look_back:i, 0])
        X_test = np.array(X_test)
        print("Shape of X_test:", X_test.shape)

        if X_test.shape[0] > 0:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        else:
            print("X_test is empty")

        predicted_data = model.predict(X_test)
        predicted_data = scaler.inverse_transform(np.reshape(predicted_data, (predicted_data.shape[0], 1)))

        # Actual data for plotting
        actual_data = stock_data[column].values[look_back:]

        # Create data for plotting actual values
        actual_plot_data = {
            'x': list(range(len(actual_data))),
            'y': actual_data.flatten().tolist(),
            'label': f"Actual {column}"
        }

        # Create data for plotting predicted values
        predicted_plot_data = {
            'x': list(range(len(predicted_data))),
            'y': predicted_data.flatten().tolist(),
            'label': f"Predicted {column}"
        }

        plot_data_list.append(actual_plot_data)
        plot_data_list.append(predicted_plot_data)

    return plot_data_list
