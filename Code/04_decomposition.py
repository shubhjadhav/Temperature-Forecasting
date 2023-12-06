import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from statsmodels.tsa.seasonal import STL

import helperfunctions as hf  # custom function files

data_path = "../Data/"

target = 'Temperature'

temp_df = hf.make_df(data_path+"temp_st.csv")

stl = STL(temp_df[target], period=1440)
res = stl.fit()

trend = np.array(res.trend)
season = np.array(res.seasonal)
residual = np.array(res.resid)

Ft = max(0, 1 - np.var(residual)/np.var(trend + residual))
print(f"\nThe strength of trend for this data set is {round(Ft, 4)} or {round(Ft*100, 2)}%")

Fs = max(0, 1 - np.var(residual)/np.var(season + residual))
print(f"\nThe strength of seasonality for this data set is {round(Fs, 4)} or {round(Fs*100, 2)}%")

res.plot()
plt.show()

plt.figure(figsize=(16, 8))
plt.plot(temp_df.index, trend, label='trend')
plt.plot(temp_df.index, season, label='Seasonal')
plt.plot(temp_df.index, residual, label='Residuals')
plt.xlabel('Year')
plt.ylabel(target)
plt.title('STL Decomposition')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
