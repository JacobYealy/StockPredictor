from flask import Flask, jsonify, render_template
import json
from lstm_model import generate_plot_data

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_plot_data')
def get_plot_data():
    plot_data_list = generate_plot_data()
    return jsonify(json.dumps(plot_data_list))

if __name__ == '__main__':
    app.run(debug=True)
