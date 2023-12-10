import seaborn as sns
import matplotlib.pyplot as plt

import helperfunctions as hf  # custom function files

data_path = "../Data/"

target = 'Temperature'

temp_df = hf.make_df(data_path+"temp_data.csv", disp_samp=1)

min_date = temp_df.index.min()
max_date = temp_df.index.max()

print(f"\nEarliest Date: {min_date}")
print(f"Latest Date: {max_date}")

temp_df[target].plot()
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Target to predict: " + target)
plt.tight_layout()
plt.grid()
plt.show()

for i, col in enumerate(temp_df.columns):
    if col == 'Occupancy': continue
    temp_df[col].plot()
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.title(col)
    plt.tight_layout()
    plt.grid()
    plt.show()

print(temp_df.info())

hf.print_tab(temp_df.describe())

temp = temp_df.isnull().sum().reset_index()
temp.columns = ['Variables', 'NA Count']
hf.print_tab(temp_df.isnull().sum().reset_index())

cor = temp_df.corr()
sns.heatmap(data = cor, annot = True, vmin=-1, vmax=1)
plt.title("Correlation Plot of all variables")
plt.tight_layout()
plt.show()

hf.plot_autocorr(temp_df[target].values, lags=20, title=f"ACF Plot for {target}")
hf.plot_acf_pcf(temp_df[target].values, lags=20, title=f"ACF/PACF Plot for {target}")
