import pandas as pd
from tabulate import tabulate
import helperfunctions as hf  # custom function files

data_path = "../Data/"

temp_df = pd.read_csv(data_path+"raw_data.csv", index_col=0)

date = pd.date_range(
    start="2015-02-11 14:48",
    end='2015-02-18 09:19',
    freq='min'
)
temp_df['date'] = date
temp_df.index = temp_df['date']
temp_df.drop(columns=['date'], inplace=True)

print(f"\nTotal records: {temp_df.shape[0]}\n")
hf.print_tab(temp_df.head())
temp_df.to_csv("../Data/temp_data.csv")
