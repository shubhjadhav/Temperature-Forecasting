import helperfunctions as hf  # custom function files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

mode_data_path = "../Data/Model Data/"

target = 'Temperature'

X_train = hf.make_df(mode_data_path + 'X_train.csv', 0)
y_train = hf.make_df(mode_data_path + 'y_train.csv', 0)[target]
X_test = hf.make_df(mode_data_path + 'X_test.csv', 0)
y_test = hf.make_df(mode_data_path + 'y_test.csv', 0)[target]

summary_res = pd.DataFrame(columns=range(11))

# --------------------------------------------------------------------------------
#                                    Coefficients using Normal equation
# --------------------------------------------------------------------------------
hf.print_h("Coefficients using Normal equation")

s, d, v = np.linalg.svd(X_train, full_matrices=True)

print(f'singular values of x are {d}')
print(f'The condition number for x is {np.linalg.cond(X_train)}')

# --------------------------------------------------------------------------------
#                                    Coefficients using Normal equation
# --------------------------------------------------------------------------------
hf.print_h("Coefficients using Normal equation")

beta_hat = hf.calc_lse(X_train, y_train)
lse_df = pd.DataFrame({
    'feature': ['const'] + X_train.columns.to_list(),
    'Beta': beta_hat
})
print(f"\nThe regression model unknown coefficients using the Normal equation")
hf.print_tab(lse_df)

# --------------------------------------------------------------------------------
#                                    OLS Model with all features
# --------------------------------------------------------------------------------
hf.print_h("OLS Model with all features")

X_train = sm.add_constant(X_train, prepend=True)
X_test = sm.add_constant(X_test, prepend=True)

ols_model_all_features = hf.calc_ols(X_train, y_train)
temp_df = np.round(ols_model_all_features.params, 2).reset_index()
lse_df = pd.DataFrame({
    'feature': temp_df['index'],
    'Beta': temp_df[0]
})
print("\nThe regression model unknown coefficients using OLS method")
hf.print_tab(lse_df)

r2 = ols_model_all_features.rsquared_adj
print(f"\n\nR-squared value model with all {X_train.shape[1]} features: {r2}")
y_train_pred = ols_model_all_features.predict(X_train)
y_pred = ols_model_all_features.predict(X_test)
hf.plot_forecast(y_train, [], y_train_pred, 'Training Prediction of Temperature using OLS method',
                 y_label='Temperature', x_label="Index")
hf.plot_forecast([], y_test, y_pred, 'Forecasting Temperature using OLS method', y_label='Temperature', x_label="Index")

res_e = (y_train - y_train_pred)
pred_e = (y_test - y_pred)
error_stat_df, error_stat = hf.cal_error_stat(res_e, pred_e, r2=r2, nm='All Features')
hf.print_tab(error_stat_df)
summary_res.columns = error_stat_df.columns
summary_res.loc[len(summary_res)] = error_stat

# --------------------------------------------------------------------------------
#                                    BSR Feature Selection
# --------------------------------------------------------------------------------
hf.print_h("BSR Feature Selection")

res_df, imp_features_bsr, features_to_drop_bsr = hf.backward_stepwise_regression_fs(X_train, y_train, logs=1)

print(f"\n\nBSR | Important Features: {imp_features_bsr}")
print(f"BSR | Features to drop: {features_to_drop_bsr}")

X_train_BSR = X_train[imp_features_bsr]
X_test_BSR = X_test[imp_features_bsr]

ols_model_imp_features = sm.OLS(y_train, X_train_BSR).fit()
r2 = ols_model_imp_features.rsquared_adj
print(f"\n\nR-squared value model with all {X_train_BSR.shape[1]} features: {ols_model_imp_features.rsquared_adj}")

y_train_pred = ols_model_imp_features.predict(X_train)
y_pred = ols_model_imp_features.predict(X_test_BSR)
hf.plot_forecast(y_train, [], y_train_pred, 'Training Prediction of Temperature with BSR reduced features',
                 y_label='Temperature', x_label="Index")
hf.plot_forecast([], y_test, y_pred, 'Forecasting Temperature with BSR reduced features', y_label='Temperature',
                 x_label="Index")

res_e = (y_train - y_train_pred)
pred_e = (y_test - y_pred)
error_stat_df, error_stat = hf.cal_error_stat(res_e, pred_e, r2=r2, nm='Features from BSR')
hf.print_tab(error_stat_df)
summary_res.loc[len(summary_res)] = error_stat

# --------------------------------------------------------------------------------
#                                    VIF Feature Selection
# --------------------------------------------------------------------------------
hf.print_h("VIF Feature Selection")

imp_features_vif, features_to_drop_vif = hf.vif_fs(X_train, y_train, logs=1)

print(f"\n\nVIF | Important Features: {imp_features_vif}")
print(f"VIF | Features to drop: {features_to_drop_vif}")

X_train_VIF = X_train[imp_features_vif]
X_test_VIF = X_test[imp_features_vif]

ols_model_imp_features = sm.OLS(y_train, X_train_VIF).fit()
r2 = ols_model_imp_features.rsquared_adj
print(f"\n\nR-squared value model with all {X_train_VIF.shape[1]} features: {r2}")

y_train_pred = hf.predict_ols(ols_model_imp_features, X_train)
y_pred = ols_model_imp_features.predict(X_test_VIF)
hf.plot_forecast(y_train, [], y_train_pred, 'Training Prediction of Temperature with VIF reduced features',
                 y_label='Temperature', x_label="Index")
hf.plot_forecast([], y_test, y_pred, 'Forecasting Temperature with VIF reduced features', y_label='Temperature',
                 x_label="Index")

res_e = (y_train - y_train_pred)
pred_e = (y_test - y_pred)
error_stat_df, error_stat = hf.cal_error_stat(res_e, pred_e, r2=r2, nm='Features from VIF')
hf.print_tab(error_stat_df)
summary_res.loc[len(summary_res)] = error_stat

# --------------------------------------------------------------------------------
#                                    PCA
# --------------------------------------------------------------------------------
hf.print_h("PCA")

sc = StandardScaler()
pca = PCA(n_components='mle')

X_train_PCA = X_train.drop(columns=['const'])
X_test_PCA = X_test.drop(columns=['const'])

X_train_PCA = pca.fit_transform(X_train_PCA)
X_test_PCA = pca.fit_transform(X_test_PCA)

X_train_PCA = sm.add_constant(X_train_PCA, prepend=True)
X_test_PCA = sm.add_constant(X_test_PCA, prepend=True)

ols_model_pca_features = sm.OLS(y_train, X_train_PCA).fit()
r2 = ols_model_pca_features.rsquared_adj
print(f"\n\nR-squared value model with all {X_train_PCA.shape[1]} features: {r2}")

y_train_pred = hf.predict_ols(ols_model_pca_features, X_train_PCA)
y_pred = ols_model_pca_features.predict(X_test_PCA)
hf.plot_forecast(y_train, [], y_train_pred, 'Training Prediction of Temperature with PCA reduced features',
                 y_label='Temperature', x_label="Index")
hf.plot_forecast([], y_test, y_pred, 'Forecasting Temperature with PCA reduced features', y_label='Temperature',
                 x_label="Index")

res_e = (y_train - y_train_pred)
pred_e = (y_test - y_pred)
error_stat_df, error_stat = hf.cal_error_stat(res_e, pred_e, r2=r2, nm='Features from PCA')
hf.print_tab(error_stat_df)
summary_res.loc[len(summary_res)] = error_stat

hf.print_tab(summary_res)
