import os
import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import requests

API_KEY = "0QP1NKR7T9294YVM"
DB_NAME = "data.sqlite"

scaler = MinMaxScaler(feature_range=(0, 1))
columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def connect_to_db():
    return sqlite3.connect(DB_NAME)

def insert_stock_data(data_frame):
    conn = connect_to_db()
    data_frame.to_sql('stock_data', conn, if_exists='append', index=False)  # changed to 'append'
    conn.close()

def insert_sentiment_data(data_frame):
    # Convert problematic columns to string format
    for col in ['authors', 'topics', 'ticker_sentiment']:
        if col in data_frame:
            data_frame[col] = data_frame[col].astype(str)

    try:
        conn = connect_to_db()
        data_frame.to_sql('sentiment_data', conn, if_exists='replace', index=False)
        conn.close()
    except Exception as e:
        print(f"Error: {e}")
        print(data_frame.head())
        print("TYPES:", data_frame.dtypes)

def fetch_stock_data_from_db():
    conn = connect_to_db()
    df = pd.read_sql('SELECT * FROM stock_data', conn)
    conn.close()
    return df

def fetch_sentiment_data_from_db():
    conn = connect_to_db()
    df = pd.read_sql('SELECT * FROM sentiment_data', conn)
    conn.close()
    return df

def fetch_latest_yfinance_data(end_date, months=7):
    start_date = (end_date - timedelta(days=months * 30)).strftime('%Y-%m-%d')
    stock_data = yf.download("TSLA", start=start_date, end=end_date.strftime('%Y-%m-%d'))
    insert_stock_data(stock_data)
    return stock_data

def fetch_alpha_vantage_data():
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&time_from=20230610T0130&sort=EARLIEST&tickers=TSLA&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()['feed']
    df = pd.DataFrame(data)
    df['date'] = df['time_published'].apply(lambda x: x.split("T")[0])
    print(df.head())
    insert_sentiment_data(df)
    return df


def fetch_sentiment_data_for_period(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch sentiment data for TSLA for the given period using the Alpha Vantage API.
    """
    # Adjust the format to '%Y%m%dT%H%M'
    start_str = start_date.strftime('%Y%m%dT%H%M')
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&time_from={start_str}&sort=EARLIEST&tickers=TSLA&apikey={API_KEY}"
    response = requests.get(url)

    # Check the content of the response
    json_content = response.json()

    # Ensure the key 'feed' exists in the response
    if 'feed' not in json_content:
        print("Error: 'feed' key not found in the response. Response:", json_content)
        return pd.DataFrame()  # Return an empty DataFrame to indicate an error

    data = json_content['feed']
    df = pd.DataFrame(data)
    df['date'] = df['time_published'].apply(lambda x: x.split("T")[0])

    return df



def fetch_data_for_last_six_months():
    """Fetch sentiment data for TSLA for the last six months in bi-monthly intervals."""
    end_date = datetime.now()
    for _ in range(3):  # fetch bi-monthly data for the past six months
        start_date = end_date - timedelta(days=60)
        df = fetch_sentiment_data_for_period(start_date, end_date)
        insert_sentiment_data(df)
        end_date = start_date


def fetch_data():
    # Fetch sentiment data
    sentiment_data = fetch_sentiment_data_from_db()

    if sentiment_data.empty:
        fetch_data_for_last_six_months()
        sentiment_data = fetch_sentiment_data_from_db()
    else:
        latest_data = fetch_sentiment_data_for_period(datetime.now() - timedelta(days=7), datetime.now())
        insert_sentiment_data(latest_data)

    latest_sentiment_date = datetime.strptime(sentiment_data['date'].max(), '%Y%m%d')


    # Fetch stock data from database
    stock_data = fetch_stock_data_from_db()

    # If the latest stock date in the database is older than the latest sentiment date, fetch new stock data
    if stock_data.empty or stock_data.index[-1] < latest_sentiment_date:
        latest_stock_data = fetch_latest_yfinance_data(datetime.now(), months=6)
        insert_stock_data(latest_stock_data)
        stock_data = fetch_stock_data_from_db()

    return stock_data, sentiment_data


if __name__ == '__main__':
    fetch_data()