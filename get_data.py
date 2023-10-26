import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import requests
import time

API_KEY = "QGUA7A72532VRXDL"
DB_NAME = "data.sqlite"

scaler = MinMaxScaler(feature_range=(0, 1))
columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def connect_to_db():
    """
        Establishes a connection to the SQLite3 database.

        Returns:
            sqlite3.Connection: Database connection object.
        """
    return sqlite3.connect(DB_NAME)


def insert_stock_data(data_frame):
    """
        Inserts stock data into the 'stock_data' table in the database.

        Parameters:
            data_frame (pd.DataFrame): Stock data.
        """
    conn = connect_to_db()
    data_frame.to_sql('stock_data', conn, if_exists='replace', index=False)
    conn.close()


def insert_sentiment_data(data_frame):
    """
        Inserts sentiment data into the 'sentiment_data' table in the database.

        Parameters:
            data_frame (pd.DataFrame): Sentiment data with columns including 'authors', 'topics', and 'ticker_sentiment'.
        """
    for col in ['authors', 'topics', 'ticker_sentiment']:
        if col in data_frame:
            data_frame[col] = data_frame[col].astype(str)

    conn = connect_to_db()
    data_frame.to_sql('sentiment_data', conn, if_exists='replace', index=False)
    conn.close()


def fetch_latest_yfinance_data():
    """
        Fetches the latest stock data for Tesla (TSLA) from Yahoo Finance for the past six months.

        Returns:
            pd.DataFrame: Stock data for the specified period.
        """
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=6 * 30)).strftime('%Y-%m-%d')
    stock_data = yf.download("TSLA", start=start_date, end=end_date.strftime('%Y-%m-%d'))
    stock_data.reset_index(inplace=True)
    insert_stock_data(stock_data)
    return stock_data


def fetch_sentiment_data_for_last_six_months():
    """
        Fetches the sentiment data for Tesla (TSLA) from Alpha Vantage for the past six months.
        Alpha Vantage limits API use to 5 calls/minute, so the function sleeps to ensure we do not
        hit the rate limit. Each API call spans 1 month (start_time to end_time), and gets 50 articles
        per call.

        Returns:
            pd.DataFrame: Aggregated sentiment data for the specified period.
        """
    today = datetime.now()
    fetched_data_frames = []

    for month in range(6):  # Last six months
        start_time = (today - timedelta(days=(month + 1) * 30)).strftime('%Y%m%dT0000')
        end_time = (today - timedelta(days=month * 30)).strftime('%Y%m%dT0000')

        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&time_from={start_time}&time_to={end_time}&sort=RELEVANCE&tickers=TSLA&apikey={API_KEY}"
        response = requests.get(url)

        print(f"Fetched articles for period {start_time} to {end_time}.")

        data = response.json()['feed']
        df = pd.DataFrame(data)
        df['date'] = df['time_published'].apply(lambda x: x.split("T")[0])

        fetched_data_frames.append(df)
        time.sleep(15)  # Sleep for 15 seconds to avoid rapid-fire requests

    # Concatenate all monthly data frames into one
    return pd.concat(fetched_data_frames, ignore_index=True)


def fetch_data():
    """
        Fetches both sentiment and stock data for Tesla (TSLA).

        Returns:
            tuple: A tuple containing the stock data and sentiment data DataFrames.
        """
    sentiment_data = fetch_sentiment_data_for_last_six_months()
    insert_sentiment_data(sentiment_data)

    stock_data = fetch_latest_yfinance_data()
    insert_stock_data(stock_data)

    return stock_data, sentiment_data


if __name__ == '__main__':
    fetch_data()