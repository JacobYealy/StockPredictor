import unittest
from unittest.mock import patch
from database import insert_data, session, StockData


class TestDatabase(unittest.TestCase):

    def setUp(self):
        # Setup a sample data to test with
        self.sample_data = [
            ('2021-09-01', 150.0, 152.0, 149.0, 151.0, 5000),
            ('2021-09-02', 151.0, 153.0, 150.0, 152.0, 6000)
        ]

    @patch('database.session')
    def test_insert_data(self, mock_session):
        # Test if insert_data adds data correctly
        insert_data(self.sample_data)
        added_data = [call[0][0] for call in mock_session.add.mock_calls]
        self.assertEqual(added_data,
                         [StockData(date='2021-09-01', open=150.0, high=152.0, low=149.0, close=151.0, volume=5000),
                          StockData(date='2021-09-02', open=151.0, high=153.0, low=150.0, close=152.0, volume=6000)])


if __name__ == '__main__':
    unittest.main()
