from flask import Flask, jsonify, render_template, redirect, url_for
from lstm_model import generate_plot_data

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


# Add form submission utils.
@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    # Handle form data and do something (e.g., store in the database, send an email, etc.)
    return redirect(url_for('home'))  # Could set up email.js for simplicity


@app.route('/get_plot_data')
def get_plot_data_endpoint():
    plot_data_list, combined_stats, stock_stats, t_test_combined, t_test_stock = generate_plot_data()
    return jsonify({
        'plot_data': plot_data_list,
        'combined_model_statistics': combined_stats,
        'stock_only_model_statistics': stock_stats,
        't_test_combined_statistics': t_test_combined,
        't_test_stock_statistics': t_test_stock
    })


if __name__ == '__main__':
    app.run(debug=True)