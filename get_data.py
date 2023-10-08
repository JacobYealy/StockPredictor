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
DB_NAME = "data.db"

def connect_to_db():
    return sqlite3.connect(DB_NAME)

def insert_stock_data(data_frame):
    conn = connect_to_db()
    data_frame.to_sql('stock_data', conn, if_exists='replace', index=False)
    conn.close()

def insert_sentiment_data(data_frame):
    conn = connect_to_db()
    data_frame.to_sql('sentiment_data', conn, if_exists='replace', index=False)
    conn.close()

def fetch_latest_yfinance_data(months=6):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=months*30)).strftime('%Y-%m-%d')
    stock_data = yf.download("TSLA", start=start_date, end=end_date)
    return stock_data

def fetch_alpha_vantage_data():
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=TSLA&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()['feed']  # Adjust based on actual JSON structure
    df = pd.DataFrame(data)
    df['date'] = df['time_published'].apply(lambda x: x.split("T")[0])
    return df

def fetch_data(months=6):
    # If cached, load from file
    if os.path.exists(STOCK_DATA_FILE):
        stock_data = pd.read_csv(STOCK_DATA_FILE, index_col='Date')
    else:
        stock_data = fetch_latest_yfinance_data(months)

    # Fetch sentiment data
    sentiment_data = fetch_alpha_vantage_data()
    earliest_sentiment_date = sentiment_data['date'].min()

    # If earliest sentiment data is not in stock data, refetch stock data
    if earliest_sentiment_date not in stock_data.index:
        stock_data = fetch_latest_yfinance_data(months=(datetime.today() - datetime.strptime(earliest_sentiment_date, '%Y-%m-%d')).days//30)

    # Forward-fill sentiment data
    sentiment_data.set_index('date', inplace=True)
    sentiment_data = sentiment_data.reindex(stock_data.index, method='ffill')

    return stock_data, sentiment_data

if __name__ == '__main__':
    fetch_data()
