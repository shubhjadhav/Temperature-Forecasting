import pandas as pd

import helperfunctions as hf  # custom function files
import numpy as np

round_off = 3
np.random.seed(6313)
na = 4
nb = 1
mode_data_path = "../Data/Model Data/"
target = 'Temperature'

y = hf.make_df(mode_data_path + 'y_st_train.csv')[target]

coeff_df = pd.DataFrame(columns=[' ', 'Coefficient'])

theta, sse, var_error, covariance_theta_hat, sse_list = hf.lm_step3(y, na, nb)

theta2 = np.array(theta).reshape(-1)

for i in range(na + nb):
    if i < na:
        coeff_df.loc[len(coeff_df)] = {
            ' ': f"AR coefficient {i + 1}",
            'Coefficient': (np.round(theta2[i], round_off))
        }
    else:
        coeff_df.loc[len(coeff_df)] = {
            ' ': f"MA coefficient {i + 1}",
            'Coefficient': (np.round(theta2[i], round_off))
        }

hf.print_tab(coeff_df)

hf.lm_confidence_interval(theta, covariance_theta_hat, na, nb, round_off=round_off)

print(f"\nEstimated Covariance Matrix of estimated parameters: \n{np.round(covariance_theta_hat, decimals=round_off)}")

print(f"Estimated variance of error: {round(var_error, round_off)}")

hf.lm_find_roots(theta, na, round_off=round_off)

hf.plot_sse(sse_list, f"ARMA({na},{nb})")
