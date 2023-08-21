from flask import Flask, jsonify, render_template
from lstm_model import generate_plot_data

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_plot_data', methods=['GET'])
def get_plot_data():
    plot_data_json = generate_plot_data()
    return jsonify({'plotData': plot_data_json})

if __name__ == '__main__':
    app.run(debug=True)
