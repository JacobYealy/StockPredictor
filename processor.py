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

    # Check the stock data after fetching
    print("Stock Data After Fetching:\n", df.head())
    return df


def fetch_sentiment_data_from_db():
    conn = connect_to_db()

    # Fetch data from both sentiment tables
    recent_sentiment_df = pd.read_sql('SELECT date, overall_sentiment_score FROM sentiment_data', conn)
    year_sentiment_df = pd.read_sql('SELECT date, overall_sentiment_score FROM sentiment_data_year', conn)
    conn.close()

    # Convert 'date' column to datetime for both dataframes
    recent_sentiment_df['date'] = pd.to_datetime(recent_sentiment_df['date'], errors='coerce')
    year_sentiment_df['date'] = pd.to_datetime(year_sentiment_df['date'], errors='coerce')

    # Combine the data from both tables
    df = pd.concat([year_sentiment_df, recent_sentiment_df]).drop_duplicates().sort_values(by='date')

    # Check the combined sentiment data after fetching
    print("Combined Sentiment Data After Fetching:\n", df.head())

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


def prepare_data(stock_data, sentiment_data=None, look_back=5, test_size=0.1):
    stock_df = pd.DataFrame(stock_data, columns=['Date', 'Close'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    # Debugging: Print shapes of input data
    print(f"Shape of stock data: {stock_data.shape}")
    if sentiment_data is not None:
        print(f"Shape of sentiment data: {sentiment_data.shape}")
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
    print("Dataset tail after normalization:\n", pd.DataFrame(data_normalized).tail())

    # Splitting the data into training and testing sets
    split_idx = int(len(data_normalized) * (1 - test_size))
    train_data, test_data = data_normalized[:split_idx], data_normalized[split_idx:]

    # Extract dates for the test set
    test_dates = merged_df.index[split_idx + look_back:]

    # Structure data for LSTM
    def create_dataset(data, look_back):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i - look_back:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)

    # Diagnostic: Print the shape of X_train, y_train, X_test, y_test
    print("Shape of X_train:", X_train.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)

    return X_train, y_train, X_test, y_test, test_dates


def main():
    # Fetch data from the database
    stock_df = fetch_stock_data_from_db()
    sentiment_df = fetch_sentiment_data_from_db()

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
    X_train, y_train, X_test, y_test, test_dates = prepare_data(stock_df, sentiment_df, look_back)

    # Diagnostic: Verify the shapes of the resulting arrays
    print("Shape of X_train:", X_train.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)


if __name__ == "__main__":
    main()
