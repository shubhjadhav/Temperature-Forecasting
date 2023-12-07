import helperfunctions as hf  # custom function files
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

mode_data_path = "../Data/Model Data/"

target = 'Temperature'

lags = 20
order = 1

y_train = hf.make_df(mode_data_path+'y_st_train.csv', 0)[target]
y_test = hf.make_df(mode_data_path+'y_st_test.csv', 0)[target]

y_train.index.freq = 'T'
y_test.index.freq = 'T'

model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=365).fit()
y_train_pred = model.predict(start=y_train.index[0], end=y_train.index[-1])
y_pred = model.forecast(len(y_test))
hf.plot_forecast(y_train, y_test, y_pred, 'Holt-Winter Method', x_label='Time', y_label='Magnitude')

res_e = y_train - y_train_pred
pred_e = y_test - y_pred

error_stat_df, _ = hf.cal_error_stat(res_e, pred_e, nm='Holt-Winter')

hf.print_tab(error_stat_df)
