import unittest
import json
from unittest.mock import patch
import pandas as pd
import lstm_model
from lstm_model import generate_plot_data


class TestLSTMFunctions(unittest.TestCase):

    @patch('yf.download')
    def test_generate_plot_data_download(self, mock_download):
        # Mock the data returned by yf.download
        mock_data = pd.DataFrame({
            'Open': [1, 2, 3, 4, 5],
            'High': [1, 2, 3, 4, 5],
            'Low': [1, 2, 3, 4, 5],
            'Close': [1, 2, 3, 4, 5],
            'Adj Close': [1, 2, 3, 4, 5],
            'Volume': [1, 2, 3, 4, 5]
        })
        mock_download.return_value = mock_data

        # Now when generate_plot_data runs yf.download, it will get mock_data
        plot_data_str = generate_plot_data()
        plot_data_list = json.loads(plot_data_str)

        self.assertIsInstance(plot_data_list, list)

    def test_generate_plot_data(self):
        plot_data_str = generate_plot_data()
        plot_data_list = json.loads(plot_data_str)

        # Checks that there is a list of dictionaries for plotting
        self.assertIsInstance(plot_data_list, list)
        for plot_data in plot_data_list:
            self.assertIsInstance(plot_data, dict)
            self.assertIn('x', plot_data)
            self.assertIn('y', plot_data)
            self.assertIn('label', plot_data)

            # Check the shape of 'x' and 'y' are consistent
            self.assertEqual(len(plot_data['x']), len(plot_data['y']))

            # Checks for at least 50 data points for enough data
            self.assertTrue(len(plot_data['y']) >= 50)

            # Check if label is one of the predefined columns
            self.assertIn(plot_data['label'], ["Open", "High", "Low", "Close", "Adj Close", "Volume"])

    def test_frontend_transfer(self):
        # Call the function and get the returned JSON string
        result_json = lstm_model.generate_plot_data()

        # Convert the JSON string back to a Python object
        result_data = json.loads(result_json)

        # Check that result_data is a list
        self.assertIsInstance(result_data, list)

        # Check that all columns are present in the results
        result_columns = [item['label'] for item in result_data]

        for column in lstm_model.columns:
            self.assertIn(column, result_columns)

        # Check that each prediction data contains 'x' and 'y' for plotting
        for item in result_data:
            self.assertIn('x', item)
            self.assertIn('y', item)

            # 'x' and 'y' should have same length
            self.assertEqual(len(item['x']), len(item['y']))

            # 'y' should contain predictions
            self.assertIsInstance(item['y'], list)
            self.assertGreater(len(item['y']), 0)

if __name__ == '__main__':
    unittest.main()
