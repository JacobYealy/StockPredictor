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

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Diagnostic: Check the stock data after fetching
    print("Stock Data After Fetching:\n", df.head())
    return df

def fetch_sentiment_data_from_db():
    conn = connect_to_db()
    df = pd.read_sql('SELECT date, overall_sentiment_score FROM sentiment_data', conn)
    conn.close()

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Diagnostic: Check the sentiment data after fetching
    print("Sentiment Data After Fetching:\n", df.head())
    return df


def aggregate_sentiment_data(sentiment_df):
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    sentiment_df.set_index('date', inplace=True)

    # Calculate mean sentiment score for each date
    aggregated_sentiment = sentiment_df.groupby(sentiment_df.index).mean()

    # Forward fill the missing dates
    aggregated_sentiment = aggregated_sentiment.resample('D').ffill()

    # Reset index to turn 'date' back into a column
    aggregated_sentiment.reset_index(inplace=True)

    return aggregated_sentiment


def prepare_data(stock_data, sentiment_data=None, look_back=5):
    stock_df = pd.DataFrame(stock_data, columns=['Date', 'Close'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    if sentiment_data is not None:
        sentiment_df = pd.DataFrame(sentiment_data, columns=['date', 'overall_sentiment_score'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

        # Aggregate sentiment data
        aggregated_sentiment = aggregate_sentiment_data(sentiment_df)

        # Merge stock and aggregated sentiment data
        merged_df = pd.merge(stock_df, aggregated_sentiment, how='left', left_on='Date', right_on='date')
        merged_df.drop('date', axis=1, inplace=True)
        merged_df.set_index('Date', inplace=True)

        merged_df['overall_sentiment_score'].fillna(method='bfill', inplace=True)  # Backward fill
        merged_df['overall_sentiment_score'].fillna(method='ffill', inplace=True)  # Forward fill

        # Diagnostic: Print dataset heads after merge and forward fill
        print("Dataset head after merge and forward fill:\n", merged_df.head())

    else:
        merged_df = stock_df

    # Normalize the data
    data_normalized = stock_scaler.fit_transform(merged_df[['Close']])
    if 'overall_sentiment_score' in merged_df.columns:
        sentiment_normalized = sentiment_scaler.fit_transform(merged_df[['overall_sentiment_score']])
        data_normalized = np.hstack((data_normalized, sentiment_normalized))

    # Diagnostic: Print dataset heads after normalization
    print("Dataset head after normalization:\n", pd.DataFrame(data_normalized).tail())

    # Structure data for LSTM
    X, y = [], []
    for i in range(look_back, len(data_normalized)):
        X.append(data_normalized[i - look_back:i])
        y.append(data_normalized[i, 0])
    X, y = np.array(X), np.array(y)

    # Diagnostic: Print the shape of X and y
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    return X, y


def main():
    # Fetch data from the database
    stock_df = fetch_stock_data_from_db()
    sentiment_df = fetch_sentiment_data_from_db()

    # Convert the date columns to datetime
    stock_df['date'] = pd.to_datetime(stock_df['Date'])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    #Somewhere from here.

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