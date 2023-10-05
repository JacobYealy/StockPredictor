from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Initialize MinMaxScaler (scale data between 0-1)
scaler = MinMaxScaler(feature_range=(0, 1))

def prepare_data(stock_data, sentiment_data=None, look_back=5):
    data = stock_data
    if sentiment_data is not None:
        # Assuming both stock_data and sentiment_data are of the same length and aligned by date
        data = np.hstack((stock_data, sentiment_data))

    data_normalized = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(data_normalized)):
        X.append(data_normalized[i - look_back:i])
        y.append(data_normalized[i, 0])  # Predicting the stock value
    X, y = np.array(X), np.array(y)
    return X, y

def train_lstm_model(X_train, y_train, epochs=50, batch_size=32):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def generate_plot_data(stock_data, sentiment_data=None, look_back=5, epochs=50, batch_size=32):
    # Prepare data for LSTM
    X_train, y_train = prepare_data(stock_data, sentiment_data, look_back)

    # Train LSTM model
    model = train_lstm_model(X_train, y_train, epochs, batch_size)

    # Predict using the trained model
    predicted_data = model.predict(X_train)
    predicted_data = scaler.inverse_transform(predicted_data)

    # Inverse transform y_train to original scale for plotting
    actual_data = scaler.inverse_transform(y_train.reshape(-1, 1))

    # Create data for plotting
    actual_plot_data = {
        'x': list(range(len(actual_data))),
        'y': actual_data.flatten().tolist(),
        'label': "Actual Stock Value"
    }
    predicted_plot_data = {
        'x': list(range(len(predicted_data))),
        'y': predicted_data.flatten().tolist(),
        'label': "Predicted Stock Value"
    }

    return [actual_plot_data, predicted_plot_data]

if __name__ == "__main__":
    # Sample data
    stock_data = np.random.rand(100, 1)
    sentiment_data = np.random.rand(100, 1)

    plot_data_list = generate_plot_data(stock_data, sentiment_data=sentiment_data)
    print(plot_data_list)