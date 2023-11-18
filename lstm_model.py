import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from processor import prepare_data, fetch_stock_data_from_db, fetch_sentiment_data_from_db

# Database configuration
DB_NAME = 'data.sqlite'

# Initialize MinMaxScaler (scale data between 0-1)
stock_scaler = MinMaxScaler(feature_range=(0, 1))
sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
look_back = 5


def train_lstm_model(X_train, y_train, epochs=100, batch_size=64):
    """
    Builds and trains the LSTM model.
    Uses ADAM optimizer and MSE loss function.

    Parameters:
        X_train: Training data.
        y_train: Target variable for the training set.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.

    Returns:
        Model: Trained LSTM model.
    """

    # Increase model complexity
    model = Sequential()
    model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))

    optimizer = Adam(learning_rate=0.0005)  # Reduced learning rate
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


def generate_plot_data():
    """
    Generates data for plotting actual and predicted stock values.

    This function fetches stock and sentiment data from the database, preprocesses the data,
    trains two LSTM models (one with sentiment and one without), makes predictions, and
    prepares the data for plotting on the frontend.

    Returns:
        list: List of dictionaries containing actual and predicted stock values.
    """
    # Fetch and prepare data
    stock_df = fetch_stock_data_from_db()
    sentiment_df = fetch_sentiment_data_from_db()

    # Fit the scalers on the respective datasets
    stock_scaler.fit(stock_df[['Close']])
    sentiment_scaler.fit(sentiment_df[['overall_sentiment_score']])

    # Preserve the dates for later use
    preserved_dates = pd.to_datetime(stock_df['Date']).dt.strftime('%Y-%m-%d').tolist()

    # Prepare data for LSTM
    X_combined, y_combined = prepare_data(stock_df, sentiment_df, look_back)
    X_stock, y_stock = prepare_data(stock_df, None, look_back)

    # Train LSTM models with more epochs and larger batch size
    model_combined = train_lstm_model(X_combined, y_combined, epochs=150, batch_size=128)
    model_stock = train_lstm_model(X_stock, y_stock, epochs=150, batch_size=128)

    # Generate predictions
    predicted_combined = model_combined.predict(X_combined)
    predicted_stock = model_stock.predict(X_stock)

    # Rescale predictions using the fitted stock scaler
    predicted_combined_rescaled = stock_scaler.inverse_transform(predicted_combined)
    predicted_stock_rescaled = stock_scaler.inverse_transform(predicted_stock)

    # Denormalize the actual y values using stock scaler and reshaping
    actual_data_rescaled = stock_scaler.inverse_transform(y_combined.reshape(-1, 1))

    # Align predictions with dates and convert values to standard Python floats for JSON serialization
    plot_data_list = []
    for actual, predicted_combined, predicted_stock, date in zip(actual_data_rescaled, predicted_combined_rescaled,
                                                                 predicted_stock_rescaled, preserved_dates):
        plot_data_list.append({
            'date': date,
            'actual': float(actual[0]),  # Convert to standard Python float
            'predicted_with_sentiment': float(predicted_combined[0]),  # Convert to standard Python float
            'predicted_stock_only': float(predicted_stock[0])  # Convert to standard Python float
        })

    return plot_data_list


if __name__ == "__main__":
    # Fetch data from the database
    stock_df = fetch_stock_data_from_db()
    sentiment_df = fetch_sentiment_data_from_db()

    # Convert to appropriate data types
    stock_df['date'] = pd.to_datetime(stock_df['Date'])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    sentiment_df['overall_sentiment_score'] = pd.to_numeric(sentiment_df['overall_sentiment_score'], errors='coerce')

    # Resample to weekly frequency
    stock_df_weekly = stock_df.resample('W-Mon',
                                        on='date').last()
    sentiment_df_weekly = sentiment_df.groupby('date').agg({'overall_sentiment_score': 'sum'}).resample('W-Mon').sum()

    # Merge dataframes
    merged_df = pd.merge(stock_df_weekly, sentiment_df_weekly, on='date', how='inner')

    # Extract required columns for model processing
    stock_data = merged_df[['Close']].values
    sentiment_data = merged_df[['overall_sentiment_score']].values

    plot_data_list = generate_plot_data()
    print(plot_data_list)