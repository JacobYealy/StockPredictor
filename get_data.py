import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import requests
from math import ceil
import time

API_KEY = "0QP1NKR7T9294YVM"
DB_NAME = "data.sqlite"

scaler = MinMaxScaler(feature_range=(0, 1))
columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


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

        # Load existing sentiment data to check for duplicates
        existing_data = pd.read_sql('SELECT * FROM sentiment_data', conn)

        # Merge the new data with the existing data and drop duplicates
        combined_data = pd.concat([existing_data, data_frame]).drop_duplicates().reset_index(drop=True)

        # Clear the old table and insert the combined data
        cur = conn.cursor()
        cur.execute("DELETE FROM sentiment_data")
        combined_data.to_sql('sentiment_data', conn, if_exists='append', index=False)

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
    stock_data.reset_index(inplace=True)  # This line will convert the date index into a column
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


def fetch_sentiment_data_for_period(initial_start_date: datetime, final_end_date: datetime) -> pd.DataFrame:
    """
    Fetch sentiment data for TSLA in roughly monthly intervals between initial_start_date and final_end_date.
    """
    all_data = []

    total_days = (final_end_date - initial_start_date).days
    intervals = ceil(total_days / 30)  # Roughly splitting the period into monthly intervals

    for i in range(intervals):
        current_start_date = initial_start_date + timedelta(days=i * 30)
        current_end_date = current_start_date + timedelta(days=30)

        if current_end_date > final_end_date:
            current_end_date = final_end_date

        # Adjust the format to '%Y%m%dT%H%M'
        start_str = current_start_date.strftime('%Y%m%dT%H%M')
        end_str = current_end_date.strftime('%Y%m%dT%H%M')

        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&time_from={start_str}&time_to={end_str}&sort=EARLIEST&tickers=TSLA&apikey={API_KEY}"

        response = requests.get(url)

        # Check the content of the response
        json_content = response.json()

        # Ensure the key 'feed' exists in the response
        if 'feed' not in json_content:
            print("Error: 'feed' key not found in the response. Response:", json_content)
            continue

        data = json_content['feed']
        all_data.extend(data)

        # Log the number of articles fetched
        print(f"Fetched {len(data)} articles for period {start_str} to {end_str}.")

        time.sleep(12)  # 5 requests per minute translates to 12 seconds between requests

    df = pd.DataFrame(all_data)
    df['date'] = df['time_published'].apply(lambda x: x[:8])

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

    # Drop rows where 'date' is NaN and convert the remaining to string type
    sentiment_data = sentiment_data.dropna(subset=['date'])
    sentiment_data['date'] = sentiment_data['date'].astype(str)

    # Check for unexpected values in the 'date' column
    try:
        latest_sentiment_date = datetime.strptime(sentiment_data['date'].max(), '%Y%m%d')
    except ValueError:
        print("Unexpected values found in the 'date' column:", sentiment_data['date'].unique())
        raise

    # Fetch stock data from database
    stock_data = fetch_stock_data_from_db()

    # If the stock data frame is empty or its latest date is older than the latest sentiment date, fetch new stock data
    latest_stock_date = datetime.strptime(stock_data['Date'].iloc[-1], '%Y-%m-%d %H:%M:%S') if not stock_data.empty else None

    if not latest_stock_date or latest_stock_date < latest_sentiment_date:
        latest_stock_data = fetch_latest_yfinance_data(datetime.now(), months=6)
        insert_stock_data(latest_stock_data)
        stock_data = fetch_stock_data_from_db()

    return stock_data, sentiment_data


if __name__ == '__main__':
    fetch_data()