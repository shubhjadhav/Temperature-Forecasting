import pandas as pd
import matplotlib.pyplot as plt
import helperfunctions as hf  # custom function files
import statsmodels.api as sm

mode_data_path = "../Data/Model Data/"
target = 'Temperature'

y_train = hf.make_df(mode_data_path + 'y_st_train.csv')[target]
y_test = hf.make_df(mode_data_path + 'y_st_test.csv')[target]

order_lst = [(10, 0, 2), (13, 0, 2), (19, 0, 3)]

pole_cancellation_obs = [
    """From residual GPAC we observe an order of ARMA(3,0)\n
    There for we add 3 to na -> there fore na = 10 + 3 = 13. And nb = 2""",
    """From residual GPAC we observe an order of ARMA(6,1)\n
    There for we add 3 to na -> there fore na = 13 + 6 = 19. And nb = 2 + 1 = 3""",
    """From residual GPAC we observe the upper traingle of GPAC table are all zeros.
    Also, the pole cancellations increases for every addition of Na and Nb.
    Even though the residuals are white and Q-value is decreasing the model is getting more complex.
    Therefore, we stop here and conclude our finalized model as AMR(10,2)"""
 ]


res_df = pd.DataFrame(columns=range(10))

for idx, order in enumerate(order_lst):

    hf.print_h(f"ARIMA{order}")

    model = sm.tsa.SARIMAX(y_train, order=order)
    model_fit = model.fit()
    print(model_fit.summary())

    # simulate forecast function and compute errors
    y_train_hat = model_fit.predict()
    y_pred = model_fit.forecast(steps=len(y_test))

    # Residual errors
    res_e = y_train - y_train_hat
    pred_e = y_test - y_pred

    _, error_stat = hf.cal_error_stat(res_e, pred_e, nm=f"ARIMA{order}")

    if len(res_df):
        res_df.loc[len(res_df)] = error_stat
    else:
        res_df.columns = list(error_stat.keys())
        res_df.loc[len(res_df)] = error_stat

    ry = []
    for lag in range(0, 21):
        ry.append(hf.cal_autocorr(res_e, lag))
    hf.gpac(ry)

    hf.print_h(pole_cancellation_obs[idx])

hf.print_tab(res_df)
