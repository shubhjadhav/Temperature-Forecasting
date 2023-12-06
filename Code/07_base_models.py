import helperfunctions as hf  # custom function files
from tabulate import tabulate
import pandas as pd

mode_data_path = "../Data/Model Data/"

target = 'Temperature'

y_train = hf.make_df(mode_data_path+'y_st_train.csv', 0)[target]
y_test = hf.make_df(mode_data_path+'y_st_test.csv', 0)[target]

summary_res = pd.DataFrame(columns=range(10))

# --------------------------------------------------------------
# Average Method
# --------------------------------------------------------------
hf.print_h('Average Forecasting Method')

avg_train_pred, avg_forecast = hf.cal_average_forecast(y_train, y_test)
hf.plot_forecast(avg_train_pred.y, avg_forecast.y, avg_forecast.y_pred, title='Average Forecasting Method')
res_e = (y_train - avg_train_pred.y_pred)[1:]
pred_e = (y_test - avg_forecast.y_pred)
error_stat_df, error_stat = hf.cal_error_stat(res_e, pred_e, nm='Average')
hf.print_tab(error_stat_df)
summary_res.columns = list(error_stat.keys())
summary_res.loc[len(summary_res)] = error_stat

# --------------------------------------------------------------
# Naive Method
# --------------------------------------------------------------
hf.print_h('Naive Method')

nv_train_pred, nv_forecast = hf.cal_naive_forecast(y_train, y_test)
hf.plot_forecast(nv_train_pred.y, nv_forecast.y, nv_forecast.y_pred, title='Naive Method')
res_e = (y_train - nv_train_pred.y_pred)[2:]
pred_e = (y_test - nv_forecast.y_pred)
error_stat_df, error_stat = hf.cal_error_stat(res_e, pred_e, nm='Naive')
hf.print_tab(error_stat_df)
summary_res.loc[len(summary_res)] = error_stat

# --------------------------------------------------------------
# Drift Method
# --------------------------------------------------------------
hf.print_h('Drift Method')

df_train_pred, df_forecast = hf.cal_drift_forecast(y_train, y_test)
hf.plot_forecast(df_train_pred.y, df_forecast.y, df_forecast.y_pred, title='Drift Method')
res_e = (y_train - df_train_pred.y_pred)[2:]
pred_e = (y_test - df_forecast.y_pred)
error_stat_df, error_stat = hf.cal_error_stat(res_e, pred_e, nm='Drift')
hf.print_tab(error_stat_df)
summary_res.loc[len(summary_res)] = error_stat


# --------------------------------------------------------------
# SES Method
# --------------------------------------------------------------
hf.print_h('SES Method')

ses_train_pred, ses_forecast = hf.cal_ses_forecast(y_train, y_test, 0.5)
hf.plot_forecast(ses_train_pred.y, ses_forecast.y, ses_forecast.y_pred, title='Simple Exponential Smoothening Method')
res_e = (y_train - ses_train_pred.y_pred)[2:]
pred_e = (y_test - ses_forecast.y_pred)
error_stat_df, error_stat = hf.cal_error_stat(res_e, pred_e, nm='SES')
hf.print_tab(error_stat_df)
summary_res.loc[len(summary_res)] = error_stat

hf.print_tab(summary_res)
