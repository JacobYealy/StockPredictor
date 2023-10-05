import unittest
from unittest.mock import patch
from datamanager import fetch_and_store_stock_data

class TestDataManager(unittest.TestCase):

    @patch('data_manager.yf')
    @patch('data_manager.insert_data')
    def test_fetch_and_store_stock_data(self, mock_insert_data, mock_yf):
        # Mock the yf.Ticker().history() call
        mock_yf.Ticker().history.return_value = YourMockedDataFrameHere  # Replace with a mock DataFrame

        # Run the function
        fetch_and_store_stock_data('AAPL')

        # Check if insert_data was called with the correct arguments
        mock_insert_data.assert_called_with(YourExpectedFormattedData)  # Replace with the data you expect

if __name__ == '__main__':
    unittest.main()
