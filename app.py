from flask import Flask, render_template
import database
import model
from datamanager import fetch_and_store_stock_data  # Imports yfin function

app = Flask(__name__)


@app.route('/')
def home():
    # Fetch and store data into SQLite
    fetch_and_store_stock_data("AAPL")

    # ... (rest of the code)


if __name__ == '__main__':
    database.create_table()
    app.run(debug=True)
