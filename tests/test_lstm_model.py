import unittest
import numpy as np
import lstm_model

"""
    TestLSTMModel - Unit Tests for LSTM Model Operations (lstm_model.py)

    Author: Jacob Yealy

    Description:
        The TestLSTMModel class provides a suite of unit tests to verify the functionality
        of the LSTM model operations as defined in lstm_model.py.
        The tests ensure:
        - Data can be successfully retrieved from the database.
        - Data is prepared correctly for the LSTM models.
        - LSTM models are trained and make predictions.

    Each test fetches sample data and checks the results against expected outputs.

    """
class TestLSTMModel(unittest.TestCase):

    def test_fetch_stock_data_from_db(self):
        df = lstm_model.fetch_stock_data_from_db()
        self.assertIsNotNone(df)
        self.assertIn('Close', df.columns)

    def test_fetch_sentiment_data_from_db(self):
        df = lstm_model.fetch_sentiment_data_from_db()
        self.assertIsNotNone(df)
        self.assertIn('overall_sentiment_score', df.columns)

    def test_prepare_data(self):
        stock_data = np.array([[10.0], [11.0], [12.0], [11.5], [11.8], [12.2], [13.0]])
        sentiment_data = np.array([[0.5], [0.6], [0.7], [0.6], [0.65], [0.68], [0.7]])

        X_combined, y_combined = lstm_model.prepare_data(stock_data, sentiment_data)
        X_stock, y_stock = lstm_model.prepare_data(stock_data)

        # Check shapes for expected sizes after look_back
        self.assertEqual(X_combined.shape, (2, 5, 2))  # 2 samples, 5 time steps, 2 features
        self.assertEqual(X_stock.shape, (2, 5, 1))  # 2 samples, 5 time steps, 1 feature

    # Checking LSTM structure
    def test_train_lstm_model_structure(self):
        dummy_data = np.random.rand(100, 5, 1)  # 100 samples, 5 time steps, 1 feature
        dummy_target = np.random.rand(100, 1)

        model = lstm_model.train_lstm_model(dummy_data, dummy_target, epochs=1, batch_size=10)

        # Check if the model has the expected number of layers
        self.assertEqual(len(model.layers), 7)  # 3 LSTM layers, 3 Dropout layers, 1 Dense layer

    def test_generate_plot_data(self):
        plot_data_list = lstm_model.generate_plot_data()
        self.assertIsInstance(plot_data_list, list)
        self.assertEqual(len(plot_data_list), 3)  # Actual, Predicted with Sentiment, Predicted Stock Only

        for plot_data in plot_data_list:
            self.assertIn('x', plot_data)
            self.assertIn('y', plot_data)
            self.assertIn('label', plot_data)


if __name__ == "__main__":
    unittest.main()
