from tabulate import tabulate
from sklearn.model_selection import train_test_split

import helperfunctions as hf  # custom function files

data_path = "../Data/"
mode_data_path = "../Data/Model Data/"

target = 'Temperature'

temp_df = hf.make_df(data_path+"temp_data.csv")

temp_st_df = hf.make_df(data_path+"temp_st.csv")

X_train, X_test, y_train, y_test = train_test_split(
    temp_df.drop(columns=[target]),
    temp_df[target],
    shuffle=False,
    test_size=0.2
)

_, _, y_st_train, y_st_test = train_test_split(
    temp_st_df.drop(columns=[target]),
    temp_st_df[target],
    shuffle=False,
    test_size=0.2
)


def min_max_norm(series):
    sr_min = min(series)
    sr_max = max(series)
    normalized_series = (series - sr_min)/(sr_max - sr_min)
    return normalized_series


for col in X_train.columns:
    X_train[col] = min_max_norm(X_train[col])

for col in X_test.columns:
    X_test[col] = min_max_norm(X_test[col])

print(f'\nTraining set size: {X_train.shape[0]} rows and {X_train.shape[1]+1} columns')
hf.print_tab(X_train.describe())

print(f'Testing set size: {X_test.shape[0]} rows and {X_test.shape[1]+1} columns')
hf.print_tab(X_test.describe())

X_train.to_csv(mode_data_path+'X_train.csv')
X_test.to_csv(mode_data_path+'X_test.csv')
y_train.to_csv(mode_data_path+'y_train.csv')
y_test.to_csv(mode_data_path+'y_test.csv')
y_st_train.to_csv(mode_data_path+'y_st_train.csv')
y_st_test.to_csv(mode_data_path+'y_st_test.csv')
