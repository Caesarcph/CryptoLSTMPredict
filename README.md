![waving](https://capsule-render.vercel.app/api?type=waving&height=200&text=Predict%20Crypto%20Price%20With%20LSTM&fontAlign=30&fontAlignY=40&color=gradient&fontSize=30)

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)

# Announcement
The results of this model are entirely derived from deep learning predictions of historical data and do not constitute any investment advice. The author will not be responsible for any investment losses caused by open source.

# parameter
- 'ticker_symbol': By modifying this parameter, you can modify the symbol you want to predict, as long as it is a symbol in the 'Yahoo API', but the model needs to be adjusted according to different symbols.

‘future_steps’: By adjusting this parameter, you can change the number of days you want to predict, but the accuracy will drop significantly after more than 30 days. It is recommended that long term predictions directly use the historical data of the weekly period for modeling.

‘look_back’: By adjusting this parameter, you can adjust the number of data days used by the model to predict each day. Changing this parameter will have a greater impact on the model.

‘ETH_Ticker.history(period="max", interval="1d")': By adjusting this function, the date interval and frequency of the obtained data can be changed.
