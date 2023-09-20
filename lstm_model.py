from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Initialize MinMaxScaler (scale data between 0-1)
scaler = MinMaxScaler(feature_range=(0, 1))

def prepare_data(data, look_back=5):
    # Reshape and normalize data
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))

    # Prepare the training data for LSTM
    X, y = [], []
    for i in range(look_back, len(data_normalized)):
        X.append(data_normalized[i - look_back:i, 0])
        y.append(data_normalized[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

def train_lstm_model(X_train, y_train, look_back=5, epochs=50, batch_size=32):
    # Build a more complex LSTM model with dropout and Bidirectional layers
    model = Sequential()
    model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Use Adam optimizer with a learning rate
    optimizer = Adam(learning_rate=0.001)

    # Compile and train the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return model

# You can add a function to generate predictions here
# ...

# You can also add a function to generate the plots and save them as PNG files
# ...

if __name__ == "__main__":
    # Test the function (replace with real stock and sentiment data)
    stock_data = np.random.rand(100)
    sentiment_data = np.random.rand(95)  # Length should match the input after applying "look_back"
    train_lstm_model(stock_data, sentiment_data)
