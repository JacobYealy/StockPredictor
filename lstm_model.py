import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from processor import prepare_data, fetch_stock_data_from_db, fetch_sentiment_data_from_db
from scipy import stats

# Database configuration
DB_NAME = 'data.sqlite'

# Initialize MinMaxScaler (scale data between 0-1)
stock_scaler = MinMaxScaler(feature_range=(0, 1))
sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
look_back = 5


def train_lstm_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Builds and trains the LSTM model.
    Uses ADAM optimizer and MSE loss function.

    Parameters:
        X_train: Training data.
        y_train: Target variable for the training set.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.

    Returns:
        Model: Trained LSTM model.
    """
    model = Sequential()
    model.add(Bidirectional(
        LSTM(units=128, return_sequences=True),
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=128))
    model.add(Dropout(0.4))
    model.add(Dense(units=1))

    optimizer = Adam(learning_rate=0.0001502609784269073)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model with validation data
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def calculate_statistics(X_test, y_test, model):
    """
    Calculates and returns model statistics such as accuracy, mean squared error, and standard deviation.

    Parameters:
        X_test: Test data features.
        y_test: Actual target values for the test data.
        model: Trained LSTM model.

    Returns:
        dict: A dictionary containing model statistics.
    """
    predicted = model.predict(X_test)
    mse = mean_squared_error(y_test, predicted)
    accuracy = r2_score(y_test, predicted)  # R-squared as a measure of accuracy
    std_dev = np.std(predicted - y_test)

    return {
        'mean_squared_error': mse,
        'accuracy': accuracy,
        'standard_deviation': std_dev
    }


def perform_t_test(predicted_stock_only, predicted_combined):
    """
    Performs a T-test to compare the predictions of the two models.

    Parameters:
        predicted_stock_only: Predictions from the stock only model.
        predicted_combined: Predictions from the combined model.

    Returns:
        dict: A dictionary containing T-test results.
    """
    # Perform T-test using the flattened arrays
    t_statistic, p_value = stats.ttest_ind(predicted_stock_only.flatten(), predicted_combined.flatten(), equal_var=False)

    return {
        't_statistic': t_statistic,
        'p_value': p_value
    }



def generate_plot_data():
    # Fetch and prepare data
    stock_df = fetch_stock_data_from_db()
    sentiment_df = fetch_sentiment_data_from_db()

    # Fit the scalers on the respective datasets
    stock_scaler.fit(stock_df[['Close']])
    sentiment_scaler.fit(sentiment_df[['overall_sentiment_score']])

    # Prepare data for LSTM with data split
    X_combined_train, y_combined_train, X_combined_test, y_combined_test, combined_test_dates = prepare_data(stock_df, sentiment_df, look_back, test_size=0.1)
    X_stock_train, y_stock_train, X_stock_test, y_stock_test, stock_test_dates = prepare_data(stock_df, None, look_back, test_size=0.1)

    # Use the preserved test set dates for plotting
    preserved_dates = combined_test_dates  # or stock_test_dates, depending on which you're plotting

    # Train LSTM models with the training set
    model_combined = train_lstm_model(X_combined_train, y_combined_train, X_combined_test, y_combined_test, epochs=50, batch_size=128)
    model_stock = train_lstm_model(X_stock_train, y_stock_train, X_stock_test, y_stock_test, epochs=50, batch_size=128)

    # Directly use flattened predictions for T-test
    predicted_combined = model_combined.predict(X_combined_test).flatten()
    predicted_stock = model_stock.predict(X_stock_test).flatten()

    # Before T-test
    print("Samples of predicted_combined for T-test:", predicted_combined[:5])
    print("Samples of predicted_stock for T-test:", predicted_stock[:5])

    # Perform T-test on flattened predictions
    t_test_results_combined = perform_t_test(predicted_combined, y_combined_test)
    t_test_results_stock = perform_t_test(predicted_stock, y_stock_test)

    # Print T-test results
    print("T-test results for combined model:", t_test_results_combined)
    print("T-test results for stock model:", t_test_results_stock)

    # Generate predictions on the test set and flatten arrays
    predicted_combined = model_combined.predict(X_combined_test).flatten()
    predicted_stock = model_stock.predict(X_stock_test).flatten()

    # Rescale predictions using the fitted stock scaler
    predicted_combined_rescaled = stock_scaler.inverse_transform(predicted_combined.reshape(-1, 1))
    predicted_stock_rescaled = stock_scaler.inverse_transform(predicted_stock.reshape(-1, 1))

    # Denormalize the actual y values using stock scaler and reshaping
    actual_data_rescaled = stock_scaler.inverse_transform(y_combined_test.reshape(-1, 1))

    # Align predictions with dates and convert values to standard Python floats for JSON serialization
    plot_data_list = []
    for actual, predicted_combined, predicted_stock, date in zip(actual_data_rescaled, predicted_combined_rescaled, predicted_stock_rescaled, preserved_dates):
        plot_data_list.append({
            'date': date,
            'actual': float(actual[0]),
            'predicted_with_sentiment': float(predicted_combined[0]),
            'predicted_stock_only': float(predicted_stock[0])
        })

    # Calculate statistics
    combined_stats = calculate_statistics(X_combined_test, y_combined_test, model_combined)
    stock_stats = calculate_statistics(X_stock_test, y_stock_test, model_stock)

    return plot_data_list, combined_stats, stock_stats, t_test_results_combined, t_test_results_stock
