import importlib

data_gathering = importlib.import_module('01_data_gathering')
data_preparation = importlib.import_module('02_data_preparation')
eda = importlib.import_module('03_eda')
stationarity = importlib.import_module('04_stationarity')
decomposition = importlib.import_module('05_decomposition')
data_preprocessing = importlib.import_module('06_data_preprocessing')
holt_winter = importlib.import_module('07_holt_winter')
base_models = importlib.import_module('08_base_models')
feature_reduction = importlib.import_module('09_feature_reduction')
regression = importlib.import_module('10_regression')
