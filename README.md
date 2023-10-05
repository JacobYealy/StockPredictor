# Stock Predictor
Jacob Yealy  
CSC-499  
Capstone Project

## Table of Contents:
- [Purpose](#purpose)
- [Hypothesis](#hypothesis)
- [Algorithm](#algorithm)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Testing](#testing)

## Purpose
This project aims to measure the effect that sentiment has on stock prediction software. 
This project utilizes the Keras library in Python to build LSTM models. 
After the LSTM models are trained and tested, one LSTM model will be given a sentiment feature, and the accuracy of each 
LSTM to true market values will be evaluated.

## Hypothesis
By integrating an emotional index with the LSTM model, 
the proposed model will be able to make more accurate predictions of Tesla stock prices compared to the model 
without the emotional index component.

## Requirements
To install all requirements, use pip install -r requirements.txt
* Keras, used to build the LSTM model
* Tensorflow, the ML library Keras is built on
* YFinance, the source of the stock data
* AlphaVantage, the source of the sentiment data.
* SQLite, the database used to store the data.
* Pandas, for dataframes.

## Dataset
* Historical Tesla stock price data: 
  * Open High, Low, Close, and Volume data
* Emotional Index data: 
  * Sentiment data from Alpha Vantage normalized as good, bad, or neutral.


## Algorithm
* Stock and sentiment data will be pulled via the yfinance and alpha vantage APIs then fitted using Pandas.
* The latest data from both APIs will be pulled. Alpha vantage will return the 50 most recent articles with sentiment scores,
and yfinance will pull 6 months of stock data so that we may use 5 contiguous months of prior data to predict the 6th month.
* The data will be preprocessed using the MinMaxScaler from SciKit Learn.
* The LSTM will be built using Keras from Tensorflow. 
* Predictions taken from the model will be transferred to the Flask application so that the data will be deployed
on a frontend web application via Chart.js.

## Testing
All tests are stored in the Test folder. To run a test, enter:
* python -m unittest + filename


