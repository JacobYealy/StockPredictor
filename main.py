from flask import Flask, render_template
import matplotlib.pyplot as plt
from get_data import fetch_yfinance_data, fetch_alpha_vantage_data
from lstm_model import train_lstm_model, prepare_data
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    # Fetch stock data and sentiment data
    stock_data = fetch_yfinance_data()
    sentiment_data = fetch_alpha_vantage_data()

    # Train the model without sentiment data
    model_without_sentiment = train_lstm_model(stock_data)

    # Prepare test data for prediction
    X_test, _ = prepare_data(stock_data)

    # Generate prediction
    predictions_without_sentiment = model_without_sentiment.predict(X_test)

    # Train the model with sentiment data
    model_with_sentiment = train_lstm_model(stock_data, sentiment_data)

    # Generate prediction with sentiment
    predictions_with_sentiment = model_with_sentiment.predict(X_test)

    # Create a plot
    plt.figure()
    plt.plot(stock_data, label='Actual Values')
    plt.plot(predictions_without_sentiment, label='Predictions without Sentiment')
    plt.plot(predictions_with_sentiment, label='Predictions with Sentiment')
    plt.legend()

    # Save plot as PNG
    plt.savefig("static/plot.png")

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
