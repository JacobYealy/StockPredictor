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
    print(df.head())
    return df
