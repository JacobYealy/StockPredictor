from keras_tuner import HyperModel, RandomSearch
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam
from lstm_model import prepare_data, fetch_stock_data_from_db, fetch_sentiment_data_from_db

class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(Bidirectional(
            LSTM(
                units=hp.Int('units', min_value=32, max_value=512, step=32),
                return_sequences=True),
            input_shape=self.input_shape))
        model.add(Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=True))
        model.add(Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
        model.add(Dropout(hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(units=1))

        model.compile(
            optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
            loss='mean_squared_error'
        )
        return model

def run_tuner(X_train, y_train):
    tuner = RandomSearch(
        LSTMHyperModel(input_shape=X_train.shape[1:]),
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory='my_dir',
        project_name='lstm_stock_prediction'
    )

    tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Get the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Print the best hyperparameters
    print("Best Hyperparameters:")
    print(best_hyperparameters.values)

    return best_model, tuner, best_hyperparameters

# You can also add a main block to run the tuner directly
if __name__ == "__main__":
    # Fetch data
    stock_data_df = fetch_stock_data_from_db()
    sentiment_data_df = fetch_sentiment_data_from_db()

    # Preprocess and prepare the data
    # This assumes that 'prepare_data' function takes raw DataFrame inputs and returns X_train, y_train
    X_train, y_train = prepare_data(stock_data_df, sentiment_data_df)

    # Run the tuner
    best_model, tuner, best_hyperparameters = run_tuner(X_train, y_train)

    # Optionally, save the best model
    best_model.save('best_lstm_model.h5')