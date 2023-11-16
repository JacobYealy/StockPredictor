import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
    # Convert to DataFrame and set date as index for stock data
    stock_df = pd.DataFrame(stock_data, columns=['Close'])
    stock_df['date'] = pd.to_datetime(stock_df.index)
    stock_df.set_index('date', inplace=True)

    if sentiment_data is not None:
        # Create a DataFrame for sentiment data
        sentiment_df = pd.DataFrame(sentiment_data, columns=['overall_sentiment_score'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df.index)
        sentiment_df.set_index('date', inplace=True)

        # Resample sentiment data to daily frequency and interpolate missing values
        sentiment_df_daily = sentiment_df.resample('D').mean()
        sentiment_df_daily['overall_sentiment_score'] = sentiment_df_daily['overall_sentiment_score'].interpolate(method='linear')

        # Debugging: Check for NaNs in stock and sentiment data before merging
        print("NaNs in stock data before merging:", stock_df.isnull().values.any())
        print("NaNs in sentiment data before merging:", sentiment_df_daily.isnull().values.any())

        # Merge stock and sentiment data
        merged_df = pd.merge(stock_df, sentiment_df_daily, left_index=True, right_index=True, how='left')
        merged_df.fillna(method='ffill', inplace=True)

        # Debugging: Check for NaNs in merged data
        print("NaNs in merged data:", merged_df.isnull().values.any())
    else:
        merged_df = stock_df

    # Normalize the data
    data_normalized = stock_scaler.fit_transform(merged_df[['Close']])
    if 'overall_sentiment_score' in merged_df.columns:
        sentiment_normalized = sentiment_scaler.fit_transform(merged_df[['overall_sentiment_score']])
        data_normalized = np.hstack((data_normalized, sentiment_normalized))

    # Debugging: Check for NaNs after normalization
    print("NaNs after normalization:", np.isnan(data_normalized).any())

    # Structure data for LSTM
    X, y = [], []
    for i in range(look_back, len(data_normalized)):
        X.append(data_normalized[i - look_back:i])
        y.append(data_normalized[i, 0])
    X, y = np.array(X), np.array(y)

    # Debugging: Check the shape of X and y
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    return X, y


def main():
    # Fetch data from the database
    stock_df = fetch_stock_data_from_db()
    sentiment_df = fetch_sentiment_data_from_db()

    # Diagnostic: Display the first few rows of the data
    print("Stock Data Head:\n", stock_df.head())
    print("Sentiment Data Head:\n", sentiment_df.head())

    # Convert the date columns to datetime
    stock_df['date'] = pd.to_datetime(stock_df['Date'])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    # Diagnostic: Check the date ranges of both datasets
    print("Stock Data Date Range: ", stock_df['date'].min(), "to", stock_df['date'].max())
    print("Sentiment Data Date Range: ", sentiment_df['date'].min(), "to", sentiment_df['date'].max())

    # Diagnostic: Check for overlapping dates
    common_dates = stock_df['date'].isin(sentiment_df['date'])
    print("Common dates between stock and sentiment data:", common_dates.sum())

    # Prepare data for LSTM
    X, y = prepare_data(stock_df, sentiment_df, look_back)

    # Diagnostic: Verify the shapes of the resulting arrays
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    # Additional diagnostics can be added here as needed

if __name__ == "__main__":
    main()