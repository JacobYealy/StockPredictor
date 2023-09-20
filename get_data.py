import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Parameters
columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# Alpha Vantage API Key and Time parameters
API_KEY = "SKW1JDXETUGX5TQA"
start = "20180101T0000"
end = "20220101T0000"


def fetch_alpha_vantage_data():
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=TSLA&time_from={start}&time_to={end}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    print("AV DATA HEAD:" + df.head())
    return df


def fetch_yfinance_data():
    # Download stock data
    stock_data = yf.download("TSLA", start="2018-01-01", end="2022-01-01")

    # Resample data to get monthly averages
    stock_data_monthly = stock_data.resample('M').mean()

    # Aggregate columns to form a single feature
    stock_data_monthly['Aggregated'] = stock_data_monthly[columns].mean(axis=1)

    # Extract and reshape data
    data = stock_data_monthly['Aggregated'].values.reshape(-1, 1)

    # Normalize the data
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler
