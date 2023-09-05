from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import yfinance as yf
from sentiment_engine import fetch_alpha_vantage_data

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

        # Splitting data into training and test sets
        train_size = int(len(data_normalized) * 0.8)
        test_size = len(data_normalized) - train_size
        train, test = data_normalized[0:train_size, :], data_normalized[train_size:len(data_normalized), :]

        # Prepare the training data for LSTM
        X_train, y_train = [], []
        for i in range(look_back, len(train)):
            X_train.append(train[i - look_back:i, 0])
            y_train.append(train[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        # Compile and train the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # Prepare the test data for LSTM
        X_test, y_test = [], []
        for i in range(look_back, len(test)):
            X_test.append(test[i - look_back:i, 0])
            y_test.append(test[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Make predictions using test data
        predicted_data = model.predict(X_test)
        predicted_data = scaler.inverse_transform(np.reshape(predicted_data, (predicted_data.shape[0], 1)))

        # Inverse transform test data to original scale
        actual_test_data = scaler.inverse_transform(test[look_back:])

        # Create data for plotting actual values
        actual_plot_data = {
            'x': list(range(len(actual_test_data))),
            'y': actual_test_data.flatten().tolist(),
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
