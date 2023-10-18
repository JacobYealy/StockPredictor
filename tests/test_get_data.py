import unittest
import pandas as pd
from get_data import fetch_data, fetch_stock_data_from_db, fetch_sentiment_data_from_db
import os

DB_NAME = os.path.join(os.pardir, "data.sqlite")

class TestDataRetrievalAndStorage(unittest.TestCase):

    def test_fetch_data(self):
        """Test the fetch_data function."""
        stock_data, sentiment_data = fetch_data()

        # Assert that data is not None
        self.assertIsNotNone(stock_data)
        self.assertIsNotNone(sentiment_data)

        # Assert that data is a DataFrame
        self.assertIsInstance(stock_data, pd.DataFrame)
        self.assertIsInstance(sentiment_data, pd.DataFrame)

        # Assert that dataframes are not empty
        self.assertFalse(stock_data.empty)
        self.assertFalse(sentiment_data.empty)

    def test_stock_data_in_db(self):
        """Test that stock data can be fetched from the DB."""
        stock_data = fetch_stock_data_from_db()

        # Assert that data is a DataFrame
        self.assertIsInstance(stock_data, pd.DataFrame)

        # Assert that dataframe is not empty
        self.assertFalse(stock_data.empty)

    def test_sentiment_data_in_db(self):
        """Test that sentiment data can be fetched from the DB."""
        sentiment_data = fetch_sentiment_data_from_db()

        # Assert that data is a DataFrame
        self.assertIsInstance(sentiment_data, pd.DataFrame)

        # Assert that dataframe is not empty
        self.assertFalse(sentiment_data.empty)

if __name__ == '__main__':
    unittest.main()
