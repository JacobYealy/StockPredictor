import os
import unittest
from unittest.mock import patch, Mock
import sqlite3
import pandas as pd
import db_setup
import get_data

DB_NAME = "data_test.sqlite"
class TestGetData(unittest.TestCase):
    """
        The TestGetData class provides a suite of unit tests to verify the functionality
        of data retrieval and insertion operations as defined in get_data.py. The tests ensure that:
        - Stock data can be inserted successfully into the database.
        - Sentiment data can be fetched and transformed accurately.
        - Both sentiment and stock data can be fetched together.

        Each test initializes a fresh test database and cleans it up after execution, ensuring
        isolation and preventing side effects between test cases.

    """
    def setUp(self):
        # Before each test case, set up a new DB with tables.
        db_setup.setup_db()

    def tearDown(self):
        # Cleanup after each test by removing the test DB
        try:
            os.remove(DB_NAME)
        except:
            pass

    @patch('get_data.connect_to_db', return_value=sqlite3.connect(DB_NAME))
    def test_insert_stock_data(self, mock_db_conn):
        # Mock df for tesing purposes
        sample_data = {
            "Open": [1.0], "High": [1.2], "Low": [0.8], "Close": [1.1], "Adj Close": [1.1], "Volume": [1000000]
        }
        df = pd.DataFrame(sample_data)
        get_data.insert_stock_data(df)

        conn = sqlite3.connect(DB_NAME)
        fetched_df = pd.read_sql("SELECT * FROM stock_data", conn)
        conn.close()

        pd.testing.assert_frame_equal(fetched_df, df)

    @patch('requests.get')
    def test_fetch_sentiment_data_for_last_six_months(self, mock_get):
        mock_response = Mock()
        mock_data = {
            "feed": [
                {"time_published": "2023-04-01T10:00", "authors": "Author", "topics": "Tech", "ticker_sentiment": "Positive"}
            ]
        }
        mock_response.json.return_value = mock_data
        mock_get.return_value = mock_response

        data = get_data.fetch_sentiment_data_for_last_six_months()
        self.assertIn("date", data.columns)
        self.assertIn("2023-04-01", data["date"].values)

    @patch('get_data.fetch_sentiment_data_for_last_six_months')
    @patch('get_data.fetch_latest_yfinance_data')
    def test_fetch_data(self, mock_yf_data, mock_sentiment_data):
        mock_stock_data = pd.DataFrame({
            "Open": [1.0], "High": [1.2], "Low": [0.8], "Close": [1.1], "Adj Close": [1.1], "Volume": [1000000]
        })
        mock_sentiment_data_df = pd.DataFrame({
            "time_published": ["2023-04-01T10:00"], "authors": ["Author"], "topics": ["Tech"],
            "ticker_sentiment": ["Positive"]
        })
        mock_yf_data.return_value = mock_stock_data
        mock_sentiment_data.return_value = mock_sentiment_data_df

        stock_data, sentiment_data = get_data.fetch_data()
        self.assertIsInstance(stock_data, pd.DataFrame)
        self.assertIsInstance(sentiment_data, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
