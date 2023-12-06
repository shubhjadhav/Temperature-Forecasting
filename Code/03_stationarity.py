import numpy as np
import matplotlib.pyplot as plt

import helperfunctions as hf  # custom function files

data_path = "../Data/"
temp_data = hf.make_df(data_path+"temp_data.csv")
target = 'Temperature'

# Stationarity

print(f"\nChecking if dependable variable is stationary:")
hf.check_stationarity_init(temp_data[target], target)

print(f"\nChecking if dependable variable is stationary after first differencing:")
temp_diff_1 = hf.non_seas_diff(temp_data[target])
hf.check_stationarity_init(temp_diff_1[1:], target)

print(); print(hf.adf_cal(temp_diff_1[1:]), end='\n\n')

print(hf.kpss_test(temp_diff_1[1:]))

temp_diff_1.plot.hist()
plt.xlabel("value")
plt.title("Histogram of 1st order differencing value")
plt.show()

hf.plot_autocorr(temp_diff_1[1:].values, lags=20, title="ACF Plot for 1st order differencing value")
hf.plot_acf_pcf(temp_diff_1[1:].values, lags=20, title="ACF Plot for 1st order differencing value")

temp_data[target] = temp_diff_1

temp_data[target].plot()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Temperature over time")
plt.tight_layout()
plt.grid()
plt.show()

temp_data.dropna(inplace=True)
temp_data.to_csv(data_path+'temp_st.csv')
