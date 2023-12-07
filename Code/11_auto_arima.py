import helperfunctions as hf  # custom function files
from sktime.forecasting.arima import AutoARIMA
import pandas as pd
import statsmodels.api as sm

mode_data_path = "../Data/Model Data/"
target = 'Temperature'

y_train = hf.make_df(mode_data_path + 'y_st_train.csv')[target]
y_test = hf.make_df(mode_data_path + 'y_st_test.csv')[target]

# p -> AR of non-seasonal
# P -> AR of seasonal
# q -> MA of non-seasonal
# Q -> MA of seasonal
# d -> non-seasonal differencing
# D -> seasonal differencing

forecast = AutoARIMA(
    start_p=0,
    max_p=10,
    start_q=0,
    max_q=10,
    start_Q=0,
    max_Q=5,
    max_d=5,
    max_D=5,
    stationary=True,
    n_fits=20,
    stepwise=False
)

auto_model = forecast.fit(y_train)
print(auto_model.summary())

model = sm.tsa.SARIMAX(y_train, order=(1,0,2))
model_fit = model.fit()
y_train_hat = model_fit.predict()
y_pred = model_fit.forecast(steps=len(y_test))

# Residual errors
res_e = y_train - y_train_hat
pred_e = y_test - y_pred

res_df = pd.DataFrame(columns=range(10))
_, error_stat = hf.cal_error_stat(res_e, pred_e, nm=f"ARMA{(1,2)}")

if len(res_df):
    res_df.loc[len(res_df)] = error_stat
else:
    res_df.columns = list(error_stat.keys())
    res_df.loc[len(res_df)] = error_stat

hf.print_tab(res_df)
