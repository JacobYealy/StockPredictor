from keras_tuner import HyperModel
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping
import tensorflow as tf
from StockPredictor.processor import fetch_stock_data_from_db, fetch_sentiment_data_from_db, prepare_data


class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        tf.keras.backend.clear_session()
        self.input_shape = input_shape

    def build(self, hp):
        """
        Builds the LSTM model with hyperparameters that will be tuned.
        This includes the number of units in each LSTM layer, the dropout rate, and the learning rate.
        The seed for reproducibility is also a tunable hyperparameter.
            """
        tf.random.set_seed(hp.Int('seed', min_value=0, max_value=100, step=1))  # Set seed as a hyperparameter
        model = Sequential()
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(Bidirectional(
                LSTM(
                    units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                    return_sequences=True if i < hp.Int('num_layers', 1, 3) - 1 else False),
                input_shape=self.input_shape))
            model.add(Dropout(hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))

        model.add(Dense(units=1))
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
            loss='mean_squared_error'
        )
        return model


def run_tuner(X_train, y_train, X_test, y_test):
    """
    Sets up the hyperparameter tuner, defines the search space, and executes the search.
    It uses random search to explore different hyperparameter combinations.
    Early stopping is used to prevent over fitting during training.
    The best model and hyperparameters are outputted after the search completes.
        """
    tuner = RandomSearch(
        LSTMHyperModel(input_shape=X_train.shape[1:]),
        objective='val_loss',
        max_trials=20,
        executions_per_trial=3,
        directory='/home/jy0441/DEV',  # Change to your DIR for reproducibility
        project_name='StockPredictor'
    )

    # Early stopping to prevent over fitting
    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[stop_early])

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Get the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Print the best hyperparameters
    print("Best Hyperparameters:")
    print(best_hyperparameters.values)

    return best_model, tuner, best_hyperparameters


if __name__ == "__main__":
    # Fetch data
    stock_data_df = fetch_stock_data_from_db()
    sentiment_data_df = fetch_sentiment_data_from_db()

    # Preprocess and prepare the data
    X_train, y_train, X_test, y_test, test_dates = prepare_data(stock_data_df, sentiment_data_df)

    # Run the tuner
    best_model, tuner, best_hyperparameters = run_tuner(X_train, y_train, X_test, y_test)