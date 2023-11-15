import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import pandas as pd
from optimizer import run_tuner

# Database configuration
DB_NAME = 'data.sqlite'

# Initialize MinMaxScaler (scale data between 0-1)
stock_scaler = MinMaxScaler(feature_range=(0, 1))
sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
look_back = 5



def connect_to_db():
    """
        Establishes and returns a connection to the SQLite database.

        Returns:
            sqlite3.Connection: Database connection object.
        """
    return sqlite3.connect(DB_NAME)


def fetch_stock_data_from_db():
    conn = connect_to_db()
    df = pd.read_sql('SELECT * FROM stock_data', conn)
    conn.close()
    return df


def fetch_sentiment_data_from_db():
    conn = connect_to_db()
    df = pd.read_sql('SELECT date, overall_sentiment_score FROM sentiment_data', conn)
    conn.close()
    return df


def prepare_data(stock_data, sentiment_data=None, look_back=5):
    """
        Prepares the data for training/testing the LSTM model.

        If sentiment data is provided, it's combined with stock data.
        The data is normalized and structured for time series forecasting.

        Parameters:
            stock_data: Stock data from the stock_data table.
            sentiment_data: Sentiment data from the sentiment_data table.
            look_back (int): Number of previous time steps to use as input variables to predict the next time period.

        Returns:
            tuple: Input features (X) and target variable (y) for the LSTM model.
        """
    if sentiment_data is not None and len(sentiment_data) > 0:
        # Normalize the sentiment_data
        sentiment_data_normalized = sentiment_scaler.fit_transform(sentiment_data)
        # Use stock_scaler to normalize stock_data
        stock_data_normalized = stock_scaler.fit_transform(stock_data)
        data_normalized = np.hstack((stock_data_normalized, sentiment_data_normalized))
    else:
        # If no sentiment data, only normalize stock data
        stock_data_normalized = stock_scaler.fit_transform(stock_data)
        data_normalized = stock_data_normalized

    X, y = [], []
    for i in range(look_back, len(data_normalized)):
        X.append(data_normalized[i - look_back:i])
        y.append(data_normalized[i, 0])  # Predicting the stock value
    X, y = np.array(X), np.array(y)
    return X, y


def generate_plot_data(best_model_combined, best_model_stock):
    """
    Generates data for plotting actual and predicted stock values.
    Uses the best models returned by the tuner for predictions.

    Parameters:
        best_model_combined: The best LSTM model trained with sentiment data.
        best_model_stock: The best LSTM model trained only with stock data.

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

    # Extract required columns for model processing
    stock_data = merged_df[['Close']].values
    sentiment_data = merged_df[['overall_sentiment_score']].values

    # Get the data prepared for LSTM models
    X_combined, y_combined = prepare_data(stock_data, sentiment_data)
    X_stock, y_stock = prepare_data(stock_data)

    # Predict using the trained best models
    predicted_combined = best_model_combined.predict(X_combined)
    predicted_stock = best_model_stock.predict(X_stock)

    # Rescale the predicted data to original scale
    predicted_combined_rescaled = stock_scaler.inverse_transform(predicted_combined)
    predicted_stock_rescaled = stock_scaler.inverse_transform(predicted_stock)
    actual_data_rescaled = stock_scaler.inverse_transform(y_combined.reshape(-1, 1))

    # Use the dates from the merged dataframe as the x-values
    # Since we're using look_back, we need to adjust the start date
    date_range = merged_df['date'][look_back:].values

    # Create data for plotting
    actual_plot_data = {
        'x': merged_df['date'][look_back:].dt.strftime('%Y-%m-%d').tolist(),
        'y': actual_data_rescaled.flatten().tolist(),
        'label': "Actual Stock Value"
    }
    predicted_combined_plot_data = {
        'x': [date.strftime('%Y-%m-%d') for date in date_range],
        'y': predicted_combined_rescaled.flatten().tolist(),
        'label': "Predicted Stock Value (with Sentiment)"
    }
    predicted_stock_plot_data = {
        'x': [date.strftime('%Y-%m-%d') for date in date_range],
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
    stock_df_weekly = stock_df.resample('W-Mon', on='date').last()
    sentiment_df_weekly = sentiment_df.groupby('date').agg({'overall_sentiment_score': 'sum'}).resample('W-Mon').sum()

    # Merge dataframes
    merged_df = pd.merge(stock_df_weekly, sentiment_df_weekly, on='date', how='inner')

    # Extract required columns for model processing
    stock_data = merged_df[['Close']].values
    sentiment_data = merged_df[['overall_sentiment_score']].values

    # Prepare the data for LSTM models
    X_combined, y_combined = prepare_data(stock_data, sentiment_data, look_back)
    X_stock, y_stock = prepare_data(stock_data, look_back=look_back)

    # Run the tuner to find the best model with combined data
    best_model_combined, tuner_combined = run_tuner(X_combined, y_combined)

    # Run the tuner to find the best model with only stock data
    best_model_stock, tuner_stock = run_tuner(X_stock, y_stock)

    # Generate plot data using the best models
    plot_data_list = generate_plot_data(best_model_combined, best_model_stock)
    print(plot_data_list)
