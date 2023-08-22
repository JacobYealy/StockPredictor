# Purpose
This project aims to measure the effect that sentiment has on stock prediction software. This project utilizes the Keras library in Python to build LSTM models. After the LSTM models are trained and tested, one LSTM model will be given a sentiment feature, and the accuracy of each LSTM to true market values will be evaluated.

# Hypothesis
By integrating an emotional index with the LSTM model, the proposed model will be able to make more accurate predictions of Tesla stock prices compared to the model without the emotional index component.

# Requirements
To install all requirements, use pip install -r requirements.txt
* Keras, used to build the LSTM model
* Tensorflow, the ML library Keras is built on
* YFinance, the source of the stock data
* SQLite, the database used to store the data.

# Dataset
* Historical Tesla stock price data: 
  * Open High, Low, Close, and Volume data
* Emotional Index data: 
  * Sentiment data from Alpha Vantage normalized as good, bad, or neutral.

# Testing
All tests are stored in the Test folder. To run a test, enter:
* python -m unittest <filename>


