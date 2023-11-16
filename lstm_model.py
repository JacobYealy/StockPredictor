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


def train_lstm_model(X_train, y_train, epochs=50, batch_size=32):
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

    # Debugging: Check for NaNs in X_train and y_train
    print("NaNs in X_train:", np.isnan(X_train).any())
    print("NaNs in y_train:", np.isnan(y_train).any())

    # Debugging: Check the shape of X_train and y_train
    print("Shape of X_train:", X_train.shape)
    print("Shape of y_train:", y_train.shape)

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

    # Debugging: Add verbose to model.fit to see more details during training
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
    # Fetch data from the database
    stock_df = fetch_stock_data_from_db()
    sentiment_df = fetch_sentiment_data_from_db()

    # Convert the date columns to datetime data type
    stock_df['date'] = pd.to_datetime(stock_df['Date'])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    # Convert sentiment score to numeric (float); errors='coerce' will set any problematic conversion to NaN
    sentiment_df['overall_sentiment_score'] = pd.to_numeric(sentiment_df['overall_sentiment_score'], errors='coerce')

    # Group by date and calculate mean sentiment score for each day
    sentiment_df_grouped = sentiment_df.groupby('date').mean().reset_index()

    # Merge dataframes on date and fill NA values with the previous non-NA value (forward fill)
    merged_df = pd.merge(stock_df, sentiment_df_grouped, on='date', how='left').fillna(method='ffill')

    # Fit the scaler on the original stock data
    stock_scaler.fit(merged_df[['Close']])

    # Extract required columns for model processing
    stock_data = merged_df[['Close']].values
    sentiment_data = merged_df[['overall_sentiment_score']].values

    # Get the data prepared for LSTM models
    X_combined, y_combined = prepare_data(stock_data, sentiment_data)
    X_stock, y_stock = prepare_data(stock_data)

    # Train LSTM models
    model_combined = train_lstm_model(X_combined, y_combined)
    model_stock = train_lstm_model(X_stock, y_stock)

    # Predict using the trained models
    predicted_combined = model_combined.predict(X_combined)
    predicted_stock = model_stock.predict(X_stock)

    # Rescale the predicted data to original scale using the fitted scaler
    predicted_combined_rescaled = stock_scaler.inverse_transform(predicted_combined.reshape(-1, 1))
    predicted_stock_rescaled = stock_scaler.inverse_transform(predicted_stock.reshape(-1, 1))
    actual_data_rescaled = stock_scaler.inverse_transform(y_combined.reshape(-1, 1))

    # Use the dates from the merged dataframe as the x-values
    date_range = merged_df['date'][look_back:].values

    # Create data for plotting
    actual_plot_data = {
        'x': merged_df['date'][look_back:].dt.strftime('%Y-%m-%d').tolist(),
        'y': actual_data_rescaled.flatten().tolist(),
        'label': "Actual Stock Value"
    }
    predicted_combined_plot_data = {
        'x': date_range.tolist(),
        'y': predicted_combined_rescaled.flatten().tolist(),
        'label': "Predicted Stock Value (with Sentiment)"
    }
    predicted_stock_plot_data = {
        'x': date_range.tolist(),
        'y': predicted_stock_rescaled.flatten().tolist(),
        'label': "Predicted Stock Value (Stock Only)"
    }

    return [actual_plot_data, predicted_combined_plot_data, predicted_stock_plot_data]


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