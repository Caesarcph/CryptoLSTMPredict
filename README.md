![waving](https://capsule-render.vercel.app/api?type=waving&height=200&text=Predict%20Crypto%20Price%20With%20LSTM&fontAlign=30&fontAlignY=40&color=gradient&fontSize=30)

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)

# Requirements
numpy==1.26.3

pandas==2.2.0

yfinance==0.2.37

matplotlib==3.8.2

scikit-learn==1.4.1.post1

tensorflow==2.15.0

# Announcement
The results of this model are entirely derived from deep learning predictions of historical data and do not constitute any investment advice. The author will not be responsible for any investment losses caused by open source.

# parameter
- `ticker_symbol`: By modifying this parameter, you can modify the symbol you want to predict, as long as it is a symbol in the 'Yahoo API', but the model needs to be adjusted according to different symbols.

- `future_steps`: By adjusting this parameter, you can change the number of days you want to predict, but the accuracy will drop significantly after more than 30 days. It is recommended that long term predictions directly use the historical data of the weekly period for modeling.

- `look_back`: By adjusting this parameter, you can adjust the number of data days used by the model to predict each day. Changing this parameter will have a greater impact on the model.

- `ETH_Ticker.history(period="max", interval="1d")`: By adjusting this function, the date interval and frequency of the obtained data can be changed.

# Model

`model = Sequential([
    LSTM(100, input_shape=(look_back, 4), return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(4) 
])`

## Model architecture

This model uses a sequence-to-sequence (Seq2Seq) architecture, which is suitable for processing time series data. The model mainly includes the following parts:

1. **LSTM Layer**: The first layer is a Long Short-Term Memory (LSTM) layer with 100 units. This layer is configured to return sequences so that the output of each time step can be passed to the next layer. It accepts input data of shape `(look_back, 4)`, where `look_back` is the number of time steps and 4 is the number of features.

2. **Dropout layer**: A Dropout layer is added after the first LSTM layer with a dropout rate of 0.2. This helps prevent the model from overfitting during training.

3. **Second LSTM layer**: This is followed by another LSTM layer with 50 units. This layer does not return the sequence, but only returns the output at the end of the sequence, which helps the model compress time series information.

4. **Second Dropout layer**: The second Dropout layer is also set with a drop rate of 0.2.

5. **Fully connected layer**: This is followed by a fully connected layer (Dense layer) with 25 units, and the activation function is ReLU. This layer uses L2 regularization with a regularization coefficient of 0.01 to further control the model complexity.

6. **Output layer**: The last layer is a Dense layer with 4 output units, corresponding to the prediction results of the model.

## Compile model

`model.compile(loss='mean_squared_error', optimizer='adam')`

The model uses mean squared error (MSE) as the loss function and uses the Adam optimizer for parameter optimization.


## Training Control

`early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)`

Use the Early Stopping strategy to prevent overfitting, monitor the loss value during the training process, and automatically stop training and retain the best model if the loss does not improve within 10 consecutive training cycles.
