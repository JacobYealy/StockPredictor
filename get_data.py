import os
import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import requests

scaler = MinMaxScaler(feature_range=(0, 1))
columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
API_KEY = "SKW1JDXETUGX5TQA"
DB_NAME = "data.sqlite"


def connect_to_db():
    return sqlite3.connect(DB_NAME)

def insert_stock_data(data_frame):
    conn = connect_to_db()
    data_frame.to_sql('stock_data', conn, if_exists='replace', index=False)
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

def fetch_latest_yfinance_data(end_date, months=6):
    start_date = (end_date - timedelta(days=months * 30)).strftime('%Y-%m-%d')
    stock_data = yf.download("TSLA", start=start_date, end=end_date.strftime('%Y-%m-%d'))
    insert_stock_data(stock_data)
    return stock_data

def fetch_alpha_vantage_data():
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=TSLA&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()['feed']
    df = pd.DataFrame(data)
    df['date'] = df['time_published'].apply(lambda x: x.split("T")[0])
    insert_sentiment_data(df)
    return df

def fetch_data():
    # Fetch sentiment data
    sentiment_data = fetch_sentiment_data_from_db()
    if sentiment_data.empty:
        sentiment_data = fetch_alpha_vantage_data()

    earliest_sentiment_date = datetime.strptime(sentiment_data['date'].min(), '%Y%m%d')
    latest_sentiment_date = datetime.strptime(sentiment_data['date'].max(), '%Y%m%d')


    # Fetch stock data from database
    stock_data = fetch_stock_data_from_db()

    # If not available in database or not aligned, fetch latest
    if stock_data.empty or earliest_sentiment_date not in stock_data.index:
        stock_data = fetch_latest_yfinance_data(latest_sentiment_date)

    return stock_data, sentiment_data

if __name__ == '__main__':
    fetch_data()