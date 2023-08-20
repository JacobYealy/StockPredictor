from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
