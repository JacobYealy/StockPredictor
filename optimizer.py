from keras_tuner import HyperModel, RandomSearch
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam

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
    # Instantiate the tuner
    tuner = RandomSearch(
        LSTMHyperModel(input_shape=X_train.shape[1:]),
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory='my_dir',
        project_name='lstm_stock_prediction'
    )

    # Start the search for the best hyperparameters
    tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model, tuner
