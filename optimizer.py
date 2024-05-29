import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from dtaidistance import dtw

def new_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0:4]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0:4])
    return np.array(dataX), np.array(dataY)

input_dim = 4
look_back = 1

def dtw_metric(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))
    
    def numpy_dtw(y_true, y_pred):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        return dtw.distance(y_true, y_pred)
    
    distance = tf.py_function(func=numpy_dtw, inp=[y_true, y_pred], Tout=tf.float32)
    return distance

def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=128, step=32),
                   input_shape=(look_back, input_dim),
                   return_sequences=True))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=32), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_3', min_value=32, max_value=128, step=32), return_sequences=False))
    model.add(Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(32, activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid']), kernel_regularizer=l2(0.01)))
    model.add(Dense(4)) 
    
    model.compile(loss=hp.Choice('loss_function', values=['mean_squared_error', 'mean_absolute_error', 'huber_loss']),
                  optimizer=hp.Choice('optimizer', values=['adam', 'nadam', 'rmsprop', 'sgd']),
                  metrics=[dtw_metric])
    
    return model

# Hyperparameter tuning
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective=kt.Objective('val_dtw_metric', direction='min'),
    max_trials=10,
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='lstm_hyperparam_tuning'
)

ticker_symbol = 'ETH-USD'
future_steps = 12
np.random.seed(7)

# Importing dataset
ETH_Ticker = yf.Ticker(ticker_symbol)
ETH_Data = ETH_Ticker.history(period="max",interval="1wk")
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = ETH_Data.iloc[:-future_steps, 0:4].values
scaled_data = scaler.fit_transform(dataset)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_dtw_metric', patience=10, verbose=1)

# Function to generate trainX and trainY based on current hyperparameters
def run_tuner_search(tuner, scaled_data, early_stopping):
    trainX, trainY = new_dataset(scaled_data, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], input_dim))
    tuner.search(x=trainX, y=trainY, epochs=250, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

run_tuner_search(tuner, scaled_data, early_stopping)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters and train it
trainX, trainY = new_dataset(scaled_data, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], input_dim))

model = build_model(best_hps)
history = model.fit(trainX, trainY, epochs=250, batch_size=32, validation_split=0.1, verbose=2, callbacks=[early_stopping])

predictedValues = np.array([])
currentStep = trainX[-1:]

for i in range(future_steps):
    nextStep = model.predict(currentStep)
    nextStepReshaped = nextStep.reshape(1, look_back, input_dim)
    predictedValues = np.append(predictedValues, nextStep)
    trainX = np.append(trainX, nextStepReshaped, axis=0)
    trainY = np.append(trainY, nextStep, axis=0)
    model.fit(trainX, trainY, epochs=2, batch_size=32, verbose=2)
    currentStep = nextStepReshaped

predictedValues = np.array(predictedValues).reshape(-1, 4)
result = scaler.inverse_transform(predictedValues)
result = pd.DataFrame(result, columns=['Open', 'High', 'Low', 'Close'])
plt.figure(figsize=(16, 8))
plt.plot(ETH_Data.iloc[-future_steps:, 0:4], label='Actual', color='blue')
ETH_Data.iloc[-future_steps:, 0:4] = result
plt.plot(ETH_Data.iloc[-future_steps:, 0:4], label='Predicted', color='red')
plt.plot(ETH_Data.iloc[:-future_steps, 0:4], label='Training', color='green')
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend(loc='lower right')
plt.show()
