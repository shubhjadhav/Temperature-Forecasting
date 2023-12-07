import pandas as pd
import matplotlib.pyplot as plt
import helperfunctions as hf  # custom function files
import statsmodels.api as sm

mode_data_path = "../Data/Model Data/"
target = 'Temperature'

y_train = hf.make_df(mode_data_path + 'y_st_train.csv')[target]
y_test = hf.make_df(mode_data_path + 'y_st_test.csv')[target]

steps = [50, 100, 150, 200]

model = sm.tsa.SARIMAX(y_train, order=(10, 0, 2))
model_fit = model.fit()

for step in steps:
    y_pred = model_fit.forecast(steps=step)
    hf.plot_forecast([], y_test[:step], y_pred, f'Test and {step}-Step Predictions for ARMA(10,2)')

