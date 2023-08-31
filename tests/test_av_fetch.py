import unittest
import json
from unittest.mock import patch
import pandas as pd
import sentiment_engine
from lstm_model import generate_plot_data


class TestDataFetch(unittest.TestCase):

    @patch('fetch_alpha_vantage_data')
    def test_data_download(self, mock_download):


    @patch('scale_transform')
    def test_sentiment_scores(self, mock_download):

