import yfinance as yf
from database import insert_data

def fetch_and_store_stock_data(stock_ticker):
    stock_data = yf.Ticker(stock_ticker)
    hist_data = stock_data.history(period="5d")
    formatted_data = [(str(row.Index), row.Open, row.High, row.Low, row.Close, row.Volume) for index, row in hist_data.iterrows()]
    insert_data(formatted_data)
