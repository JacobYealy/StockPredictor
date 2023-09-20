import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Parameters
columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# Alpha Vantage API Key and Time parameters
API_KEY = "SKW1JDXETUGX5TQA"
start = "20210101T0400" # T = time 0400 = 4:00
end = "20230601T0400"

# Convert Alpha Vantage timestamp to YYYY-MM-DD
def convert_alpha_vantage_timestamp(alpha_vantage_timestamp):
    date_str = alpha_vantage_timestamp[:8]
    return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')

def fetch_alpha_vantage_data():
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=TSLA&time_from={start}&time_to={end}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    return df


def fetch_yfinance_data():
    # Download stock data
    stock_data = yf.download("TSLA", start="2021-01-01", end="2023-06-01")

    # Resample data to get monthly averages
    stock_data_monthly = stock_data.resample('M').mean()

    # Aggregate columns to form a single feature
    stock_data_monthly['Aggregated'] = stock_data_monthly[columns].mean(axis=1)

    # Extract and reshape data
    data = stock_data_monthly['Aggregated'].values.reshape(-1, 1)

    # Normalize the data
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler


# Fetch and align data from both sources
def fetch_all_data():
    yfinance_data, yfinance_scaler = fetch_yfinance_data()
    yfinance_data = yfinance_data.resample('M').mean()

    alpha_vantage_data = fetch_alpha_vantage_data()
    alpha_vantage_data['date'] = alpha_vantage_data['timestamp'].apply(convert_alpha_vantage_timestamp)
    alpha_vantage_data.set_index('date', inplace=True)
    alpha_vantage_data = alpha_vantage_data.resample('M').mean()

    # Align data
    common_dates = yfinance_data.index.intersection(alpha_vantage_data.index)
    yfinance_data = yfinance_data.loc[common_dates]
    alpha_vantage_data = alpha_vantage_data.loc[common_dates]
    print("Yfin"  + yfinance_data)
    print("\n\n\nALPHA VANTAGE:" + alpha_vantage_data)

    return yfinance_data, yfinance_scaler, alpha_vantage_data
