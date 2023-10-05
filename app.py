from flask import Flask, jsonify, render_template
import json
from get_data import fetch_all_data
from lstm_model import generate_plot_data

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_plot_data')
def get_plot_data_endpoint():
    yfinance_data, yfinance_scaler, alpha_vantage_data = fetch_all_data()

    # Assuming the function is set up to use both datasets
    plot_data_list = generate_plot_data(yfinance_data, alpha_vantage_data)

    return jsonify(json.dumps(plot_data_list))


if __name__ == '__main__':
    app.run(debug=True)
