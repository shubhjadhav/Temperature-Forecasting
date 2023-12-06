import pandas as pd
import matplotlib.pyplot as plt
import helperfunctions as hf  # custom function files
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats

mode_data_path = "../Data/Model Data/"
target = 'Temperature'
lags = 30

y_train = hf.make_df(mode_data_path + 'y_st_train.csv')[target]
y_test = hf.make_df(mode_data_path + 'y_st_test.csv')[target]

order_lst = [(5, 0, 6), (8, 0, 1), (10, 0, 2), (11, 0, 7)]
# order_lst = [(5, 0, 6), (8, 0, 1)]

res_df = pd.DataFrame(columns=range(10))

for order in order_lst:

    hf.print_h(f"ARIMA{order}")

    model = sm.tsa.SARIMAX(y_train, order=order)
    model_fit = model.fit()
    print(model_fit.summary())

    # simulate forecast function and compute errors
    y_train_hat = model_fit.predict()
    hf.plot_forecast(y_train[:100], [], y_train_hat[:100], f'Train and One-Step Predictions for ARIMA{order}')

    y_pred = model_fit.forecast(steps=len(y_test))
    hf.plot_forecast([], y_test[:100], y_pred[:100], f'Test and One-Step Predictions for ARIMA{order}')

    hf.plot_forecast(y_train, y_test, y_pred, f'Train, Test and One-Step Predictions for ARIMA{order}')

    model_fit.plot_diagnostics(figsize=(14, 10))
    plt.suptitle(f'ARIMA{order} Diagnostic Analysis')
    plt.grid()
    plt.show()

    # Residual errors
    res_e = y_train - y_train_hat
    pred_e = y_test - y_pred

    _, error_stat = hf.cal_error_stat(res_e, pred_e, nm=f"ARIMA{order}")

    if len(res_df):
        res_df.loc[len(res_df)] = error_stat
    else:
        res_df.columns = list(error_stat.keys())
        res_df.loc[len(res_df)] = error_stat

hf.print_tab(res_df)
