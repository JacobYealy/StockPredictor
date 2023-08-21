from flask import Flask, render_template
from database import Session, Base, engine
from models.lstm_model import plot_data_json

# Initialize Flask app
app = Flask(__name__)

# Initialize database tables
Base.metadata.create_all(engine)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

def generate_plot_data():
    # your LSTM and plotting code
    # ...
    return plot_data_json
