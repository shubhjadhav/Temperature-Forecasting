import importlib

data_preparation = importlib.import_module('01_data_preparation')
eda = importlib.import_module('02_eda')
stationarity = importlib.import_module('03_stationarity')
decomposition = importlib.import_module('04_decomposition')
data_preprocessing = importlib.import_module('05_data_preprocessing')
holt_winter = importlib.import_module('06_holt_winter')
base_models = importlib.import_module('07_base_models')
regression = importlib.import_module('08_regression')
arma = importlib.import_module('09_ARMA')
lm = importlib.import_module('10_LM')
auto_arima = importlib.import_module('11_auto_ARIMA')
hstep = importlib.import_module('12_Hstep')
