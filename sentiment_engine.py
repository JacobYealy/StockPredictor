import requests
import numpy as np
import pandas as pd

API_KEY = "SKW1JDXETUGX5TQA"

# Time format: YYYYMMDDTHHMM (20180101T04:15)
start = "20180101T0000";
end = "20220101T0000";
# Grabs the sentiment data from alpha vantage via HTTP request
# Defines variables for specification
def fetch_alpha_vantage_data():
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=TSLA&time_from={start}&time_to={end}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df = df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. adjusted close': 'Adj Close', '6. volume': 'Volume'})
    df = df.astype(float)
    print(df.head())
    return df


# Alpha vantage by default scores the articles between 0-1 for a bearish to bullish type.
# From here, we will pull all of the sentiment scores and average them to get the overall average for prediction
# Takes the resulting pandas df and gets average to return to LSTM
def getScore(scores):