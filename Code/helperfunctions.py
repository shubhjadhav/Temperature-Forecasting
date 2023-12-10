import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
from tabulate import tabulate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
import math
from scipy import signal
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn import linear_model

warnings.simplefilter('ignore', InterpolationWarning)
warnings.filterwarnings("ignore")


def print_tab(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))


def print_h(text, center=1):
    print("\n\n" + "-"*100)
    print(" "* 40 if center else 0 + text)
    print("-" * 100 + "\n\n")


def make_df(path, disp_samp=1):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df.index = df['date']
    df.drop(columns=['date'], inplace=True)

    if disp_samp:
        print_tab(df.head())

    return df


def get_data_repo_url():
    return "https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/"


def cal_rolling_mean_var(data):
    roll_mean = []
    roll_var = []
    for i in range(0, len(data)):
        roll_mean.append(np.mean(data[:i + 1]))
        roll_var.append(np.var(data[:i + 1]))
    return roll_mean, roll_var


def plot_rol_mean_var(data, column_name, main_title=''):

    if main_title == '':
        main_title = 'Plot of Rolling Mean and Variance of {0}'.format(column_name)

    roll_mean, roll_var = cal_rolling_mean_var(data)

    f, axes = plt.subplots(2)

    axes[0].plot(np.arange(1, len(roll_mean) + 1), roll_mean, label='Varying mean')
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Rolling Mean - {}'.format(column_name))
    axes[0].legend()

    axes[1].plot(np.arange(1, len(roll_var) + 1), roll_var, label='Varying variance')
    axes[1].set_xlabel('Samples')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Rolling Variance - {}'.format(column_name))
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def adf_cal(ts):
    result = adfuller(ts)

    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')

    for key, value in result[4].items():
        print('\t%s: %f' % (key, value))


def kpss_test(ts):
    print('Results of KPSS Test:')

    kpsstest = kpss(ts, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])

    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value

    print(kpss_output)


def check_stationarity_init(ts, column_name):
    plot_rol_mean_var(ts, column_name)

    adf_p_val = adfuller(ts)[1]
    kpss_p_val = kpss(ts, regression='c', nlags="auto")[1]

    if adf_p_val < 0.05:
        print("According to ADF test, the time series is stationary")
    else:
        print("According to ADF test, the time series is not stationary")

    if kpss_p_val > 0.05:
        print("According to KPSS test, the time series is stationary")
    else:
        print("According to KPSS test, the time series is not stationary")


def non_seas_diff(ts, order=1):
    counter = 0
    while counter != order:
        ts = ts - ts.shift(periods=1)
        counter += 1
    return ts


def correlation_coefficient_cal(x, y):
    if len(x) == len(y):
        x_bar = np.mean(x)
        y_bar = np.mean(y)
        num = sum((x - x_bar) * (y - y_bar))
        deno = np.sqrt(sum(np.square(x - x_bar))) * np.sqrt(sum(np.square(y - y_bar)))
        return round(num / deno, 2)
    else:
        return 0


# Generate random numbers from a normal distribution
def generate_normal_distribution(mean=0, variance=1, num_observations=1000):
    np.random.seed(6313)
    random_numbers = np.random.normal(mean, np.sqrt(variance), num_observations)
    return random_numbers


# White noise data
def generate_white_noise(mean=0, std=1, sample_size=100):
    np.random.seed(6313)
    return np.random.normal(mean, std, sample_size)


# Auto Correlation Function
def cal_autocorr(y, lag):
    mean_y = np.mean(y)
    numerator = 0
    denominator = 0

    for t in range(0, len(y)):
        denominator += (y[t] - mean_y) ** 2

    for t in range(lag, len(y)):
        numerator += (y[t] - mean_y) * (y[t - lag] - mean_y)

    return numerator / denominator


# Auto Correlation Function graph
def plot_autocorr(y, lags, title='', show_plot=1):
    ryy = []
    ryy_final = []
    lags_final = []

    for lag in range(0, lags + 1):
        ryy.append(cal_autocorr(y, lag))

    ryy_final.extend(ryy[:0:-1])
    ryy_final.extend(ryy)
    lags = list(range(0, lags + 1, 1))
    lags_final.extend(lags[:0:-1])
    lags_final = [value * (-1) for value in lags_final]
    lags_final.extend(lags)

    markers, stem_lines, baseline = plt.stem(lags_final, ryy_final)
    plt.setp(markers, color='red', marker='o')
    plt.setp(stem_lines, color='#74aad0')
    plt.setp(baseline, color='grey', linewidth=2, linestyle='-')
    plt.axhspan((-1.96 / np.sqrt(len(y))), (1.96 / np.sqrt(len(y))), alpha=0.1, color='blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.tight_layout()

    if show_plot:
        plt.show()


def cal_average_forecast(train_data, test_data, step=1):
    train_pred_lst = [
        np.nan if i - step < 0
        else np.mean(train_data[:i - step + 1])
        for i in range(len(train_data))
    ]
    test_pred_lst = [np.mean(train_data)] * len(test_data)

    train_pred = pd.DataFrame({
        "y": train_data,
        "y_pred": np.round(train_pred_lst, 2)
    })

    test_pred = pd.DataFrame({
        "y": test_data,
        "y_pred": np.round(test_pred_lst, 2)
    })

    return train_pred, test_pred


def cal_naive_forecast(train_data, test_data, step=1):
    train_pred_lst = [np.nan if i - step < 0 else train_data[i - step] for i in range(len(train_data))]
    test_pred_lst = [train_data[len(train_data) - 1]] * len(test_data)

    train_pred = pd.DataFrame({
        "y": train_data,
        "y_pred": np.round(train_pred_lst, 2)
    })

    test_pred = pd.DataFrame({
        "y": test_data,
        "y_pred": np.round(test_pred_lst, 2)
    })

    return train_pred, test_pred


def cal_drift_forecast(train_data, test_data, step=1):
    len_train_data = len(train_data)
    len_test_data = len(test_data)

    train_pred_lst = [
        np.nan if i - step <= 0
        else train_data[i - step] + ((train_data[i - step] - train_data[0]) / (i - 1))
        for i in range(len_train_data)
    ]
    test_pred_lst = [
        train_data[len_train_data - 1] + (i + 1) * (train_data[len_train_data - 1] - train_data[0]) / (
                    len_train_data - 1)
        for i in range(len_test_data)
    ]

    train_pred = pd.DataFrame({
        "y": train_data,
        "y_pred": np.round(train_pred_lst, 2)
    })

    test_pred = pd.DataFrame({
        "y": test_data,
        "y_pred": np.round(test_pred_lst, 2)
    })

    return train_pred, test_pred


def cal_ses_forecast(train_data, test_data, alpha=0):
    len_train_data = len(train_data)
    len_test_data = len(test_data)
    train_pred_lst = []
    test_pred_lst = []

    for idx in range(len_train_data):
        if idx == 0:
            train_pred_lst.append(train_data[0])
        else:
            train_pred_lst.append((alpha * train_data[idx - 1]) + ((1 - alpha) * train_pred_lst[idx - 1]))

    for idx in range(len_test_data):
        test_pred_lst.append((alpha * train_data[len_train_data-1]) + ((1 - alpha) * train_pred_lst[-1]))

    train_pred = pd.DataFrame({
        "y": train_data,
        "y_pred": np.round(train_pred_lst, 2)
    })

    test_pred = pd.DataFrame({
        "y": test_data,
        "y_pred": np.round(test_pred_lst, 2)
    })

    return train_pred, test_pred


def plot_forecast(y_train, y_test, y_pred, title, x_label='Time', y_label='Magnitude', show_plot=1):
    train_n = len(y_train)
    test_n = len(y_test)
    pred_n = len(y_pred)
    dim = train_n

    if train_n:
        y_train.plot(label='Training dataset')

    if test_n:
        dim += test_n
        y_test.plot(label='Testing dataset', color='orange')

    if pred_n:
        if test_n:
            y_pred.plot(label='Forecasted values', color='green')
        else:
            y_pred.plot(label='Forecasted values', color='green')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if show_plot:
        plt.show()


def calc_stats(y, y_pred, skip=0):
    temp_data = pd.DataFrame({
        "y": y.to_list(),
        "y_pred": y_pred.to_list()
    })
    error_lst = y - y_pred

    temp_data['e'] = temp_data.y - temp_data.y_pred
    temp_data['e2'] = np.square(temp_data.e)

    error_w_skip = error_lst.iloc[skip:]
    mse = np.nansum(np.square(error_w_skip)) / len(error_w_skip)
    var = np.var(error_w_skip)
    mean_res_err = np.nanmean(error_w_skip)

    return temp_data, mse, var, mean_res_err


def cal_q_value(data, lags):
    auto_corr_val = [cal_autocorr(data, lag)**2 for lag in range(1, lags+1)]
    q_value = len(data) * np.sum(auto_corr_val)
    return np.round(q_value, 2)


def calc_lse(features, target):
    features = sm.add_constant(features, prepend=True)
    beta_hat = np.dot(np.linalg.inv(np.dot(features.T, features)), np.dot(features.T, target))
    return beta_hat


def calc_ols(features, target):
    model_ols = sm.OLS(target, features)
    result = model_ols.fit()
    return result


def predict_ols(model, features):
    features = sm.add_constant(features, prepend=True)
    y_pred = model.predict(features)
    return y_pred


def backward_stepwise_regression_fs(inp_features, inp_target, logs=0):
    features = inp_features.copy(deep=True)
    target = inp_target.copy(deep=True)

    model = sm.OLS(target, features).fit()
    aic, bic, adj_r2 = model.aic.round(2), model.bic.round(2), model.rsquared_adj.round(2)

    res_df = pd.DataFrame({
        "dropped_features": ["none"],
        "aic": [aic],
        "bic": [bic],
        "adj_r2": [adj_r2]
    })

    rank_df = pd.DataFrame(model.pvalues[1:], index=features.columns, columns=['p-value'])
    rank_df['imp_rank'] = rank_df['p-value'].rank(ascending=False)

    if logs:
        print(f'========Step {1}, include all {len(features.columns)-1} ========features\n')
        print(model.summary())
        print('\nAIC = ', aic.round(2))
        print('BIC = ', bic.round(2))
        print('Adj_R2 = ', adj_r2.round(2), '\n')
        print(rank_df)

    features_to_drop = []

    i = 1
    while len(features.columns) > 1:
        i += 1
        if (rank_df[rank_df.imp_rank == 1]['p-value'] < 0.05).values[0]:
            if logs:
                print("\nAll insignificant features are removed hence we stop here.\n")
                print(model.summary())
            break
        else:
            least_imp_feature = rank_df[rank_df.imp_rank == 1].index[0]
            features_to_drop.append(least_imp_feature)
            features.drop(columns=[least_imp_feature], axis=1, inplace=True)

            model = sm.OLS(target, features).fit()
            aic, bic, adj_r2 = model.aic.round(2), model.bic.round(2), model.rsquared_adj.round(4)

            rank_df = pd.DataFrame(model.pvalues[1:], index=features.columns, columns=['p-value'])
            rank_df['imp_rank'] = rank_df['p-value'].rank(ascending=False)

            res_df.loc[len(res_df)] = {
                "dropped_features": least_imp_feature,
                "aic": aic,
                "bic": bic,
                "adj_r2": adj_r2
            }
            if logs:
                print(f"\n\n========Step {i}, removing '{least_imp_feature}' and include {len(features.columns)-1} ========features\n")
                print(model.summary())
                print('\nAIC = ', aic)
                print('BIC = ', bic)
                print('Adj_R2 = ', adj_r2, '\n')
                print(rank_df)

    imp_features = features.columns.to_list()

    if logs:
        print(res_df)

    return res_df, imp_features, features_to_drop


def cal_vif(data):
    vif_data = pd.DataFrame()
    vif_data['feature'] = data.columns
    vif_data['vif'] = [round(variance_inflation_factor(data.values, i), 2) for i in range(len(data.columns))]
    vif_data['imp_rank'] = vif_data['vif'].rank(ascending=False)
    return vif_data


def vif_fs(inp_features, inp_target, logs=0):
    features = inp_features.copy(deep=True)
    target = inp_target.copy(deep=True)

    model = sm.OLS(target, features).fit()
    best_aic, best_bic, best_adj_r2 = model.aic, model.bic, model.rsquared_adj

    vif_df = cal_vif(features)

    if logs:
        print(f'\n\n========Step {1}, include all {len(features.columns)-1} ========features\n')
        print(model.summary())
        print('\nAIC = ', best_aic.round(2))
        print('BIC = ', best_bic.round(2))
        print('Adj_R2 = ', best_adj_r2.round(4), '\n')
        print(vif_df)

    features_to_drop = []

    i = 2
    while len(features.columns) > 1:

        least_imp_feature = vif_df[vif_df.imp_rank == 1].feature.values[0]
        features_to_drop.append(least_imp_feature)
        temp_features = features.drop(columns=[least_imp_feature], axis=1)

        model = sm.OLS(target, temp_features).fit()
        aic, bic, adj_r2 = model.aic, model.bic, model.rsquared_adj

        vif_df = cal_vif(temp_features)

        if logs:
            print(f"\n\n========Step {i}, removing '{least_imp_feature}' and include {len(temp_features.columns)-1} ========features\n")
            print(model.summary())
            print('\nAIC = ', aic.round(2))
            print('BIC = ', bic.round(2))
            print('Adj_R2 = ', adj_r2.round(4), '\n')
            print(vif_df)

        if aic <= best_aic or bic <= best_bic or best_adj_r2 <= adj_r2:
            best_aic = aic
            best_bic = bic
            best_adj_r2 = adj_r2
            i += 1
            features.drop(columns=[least_imp_feature], axis=1, inplace=True)
        else:
            if logs: print("\nThere is no improvement in the accuracy metrics hence we stop here.")
            features_to_drop.pop()
            break

    imp_features = features.columns.to_list()

    return imp_features, features_to_drop


def calc_moving_average(lst, ma_order):
    lst_len = len(lst)
    end_point = math.floor(ma_order / 2)
    mid_point = math.ceil(ma_order/2)
    ma_res = [np.nan]*lst_len

    if ma_order % 2 != 0:
        for i in range(mid_point, lst_len-end_point+1):
            values = lst[i-end_point-1:i+end_point]
            ma_res[i-1] = np.sum(values)/len(values)
    elif ma_order > 2:
        for i in range(mid_point, lst_len - end_point + 1):
            values = lst[i - end_point:i + end_point]
            ma_res[i-1] = np.sum(values)/len(values)
    elif ma_order == 2:
        for i in range(1, lst_len):
            values = lst[i - 1:i+1]
            ma_res[i] = np.sum(values)/len(values)

    return np.round(ma_res, 2)


def dlsim_method(n, num, den, mean=0, std=1):
    e = np.random.normal(mean, std, n)
    system = (num, den, 1)
    t, y_dlsim = signal.dlsim(system, e)
    return e.round(4), y_dlsim.reshape(-1).round(4)


def plot_acf_pcf(y, lags, title='ACF/PACF of the raw data'):
    fig = plt.figure()
    plt.subplot(211)
    plt.title(title)
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()


def get_arma_input():
    n = int(input('Number of observations: '))
    mean_e = int(input('Enter the mean of white noise: '))
    var_e = int(input('Enter the variance of white noise: '))
    lags = int(input('Enter number of lags: '))
    na = int(input('Enter AR order: '))
    nb = int(input('Enter MA order: '))
    den = []
    for i in range(1, na + 1):
        den.append(float(input(f'Enter the coefficient {i} of AR process: ')))
    num = []
    for i in range(1, nb + 1):
        num.append(float(input(f'Enter the coefficient {i} of MA process: ')))
    max_order = max(na, nb)
    ar_coef = [0] * max_order
    ma_coef = [0] * max_order
    for i in range(na):
        ar_coef[i] = den[i]
    for i in range(nb):
        ma_coef[i] = num[i]
    ar_params = np.array(ar_coef)
    ma_params = np.array(ma_coef)
    ar = np.r_[1, ar_params]
    ma = np.r_[1, ma_params]
    return n, na, nb, ar, ma, mean_e, var_e, lags


def gpac(ry, show_heatmap=1, j_max=7, k_max=7, round_off=2, seed=6313):
    np.random.seed(seed)
    gpac_table = np.zeros((j_max, k_max-1))

    for j in range(0, j_max):
        for k in range(1, k_max):
            phi_num = np.zeros((k, k))
            phi_den = np.zeros((k, k))
            for x in range(0, k):
                for z in range(0, k):
                    phi_num[x][z] = ry[abs(j + 1 - z + x)]
                    phi_den[x][z] = ry[abs(j - z + x)]
            phi_num = np.roll(phi_num, -1, 1)
            det_num = np.linalg.det(phi_num)
            det_den = np.linalg.det(phi_den)
            if det_den != 0 and not np.isnan(det_den):
                phi_j_k = det_num / det_den
            else:
                phi_j_k = np.nan
            gpac_table[j][k - 1] = phi_j_k

    if show_heatmap:
        plt.figure(figsize=(16, 8))
        x_axis_labels = list(range(1, k_max))
        sns.heatmap(gpac_table, annot=True, xticklabels=x_axis_labels, fmt=f'.{round_off}f', vmin=-0.1, vmax=0.1)
        plt.title(f'GPAC Table', fontsize=18)
        plt.show()

    return gpac_table


def arma_gpac_pacf(j_max=7, k_max=7, precision=2):
    n, na, nb, ar, ma, mean_e, var_e, lags = get_arma_input()
    print('\nAR coefficients: ', ar)
    print('MA coefficients:', ma)

    arma_process = sm.tsa.ArmaProcess(ar, ma)
    mean_y = mean_e*(1 + np.sum(ar))/(1 + np.sum(ma))
    y = arma_process.generate_sample(n, scale=np.sqrt(var_e)) + mean_y
    print('\nARMA Process:', list(np.around(np.array(y[:15]), precision)))

    ry = arma_process.acf(lags=lags)
    print('\nACF:', list(np.around(np.array(ry[:15]), precision)))

    gpac(ry, j_max=j_max, k_max=k_max, round_off=precision)
    plot_acf_pcf(y, lags=20)


def lm_cal_e(y, na, theta, seed=6313):
    np.random.seed(seed)
    den = theta[:na]
    num = theta[na:]
    if len(den) > len(num):  # matching len of num and den
        for x in range(len(den) - len(num)):
            num = np.append(num, 0)
    elif len(num) > len(den):
        for x in range(len(num) - len(den)):
            den = np.append(den, 0)
    den = np.insert(den, 0, 1)
    num = np.insert(num, 0, 1)
    sys = (den, num, 1)
    _, e = signal.dlsim(sys, y)
    return e


def lm_step1(y, na, nb, delta, theta):
    n = na + nb
    e = lm_cal_e(y, na, theta)
    sse_old = np.dot(np.transpose(e), e)
    X = np.empty(shape=(len(y), n))
    for i in range(0, n):
        theta[i] = theta[i] + delta
        e_i = lm_cal_e(y, na, theta)
        x_i = (e - e_i) / delta
        X[:, i] = x_i[:, 0]
        theta[i] = theta[i] - delta
    A = np.dot(np.transpose(X), X)
    g = np.dot(np.transpose(X), e)
    return A, g, X, sse_old


def lm_step2(y, na, A, theta, mu, g):
    delta_theta = np.matmul(np.linalg.inv(A + (mu * np.identity(A.shape[0]))), g)
    theta_new = theta + delta_theta
    e_new = lm_cal_e(y, na, theta_new)
    sse_new = np.dot(np.transpose(e_new), e_new)
    if np.isnan(sse_new):
        sse_new = 10 ** 10
    return sse_new, delta_theta, theta_new


def lm_step3(y, na, nb):
    N = len(y)
    n = na+nb
    mu = 0.01
    mu_max = 10 ** 20
    max_iterations = 500
    delta = 10 ** -6
    var_error = 0
    covariance_theta_hat = 0
    sse_list = []
    theta = np.zeros(shape=(n, 1))

    for iterations in range(max_iterations):
        A, g, X, sse_old = lm_step1(y, na, nb, delta, theta)
        sse_new, delta_theta, theta_new = lm_step2(y, na, A, theta, mu, g)
        sse_list.append(sse_old[0][0])
        if iterations < max_iterations:
            if sse_new < sse_old:
                if np.linalg.norm(np.array(delta_theta), 2) < 10 ** -3:
                    theta_hat = theta_new
                    var_error = sse_new / (N - n)
                    covariance_theta_hat = var_error * np.linalg.inv(A)
                    print(f"\nConvergence Occured in {iterations} iterations")
                    break
                else:
                    theta = theta_new
                    mu = mu / 10
            while sse_new >= sse_old:
                mu = mu * 10
                if mu > mu_max:
                    print('\nNo Convergence')
                    break
                sse_new, delta_theta, theta_new = lm_step2(y, na, A, theta, mu, g)
        if iterations > max_iterations:
            print('\nMax Iterations Reached')
            break
        theta = theta_new
    return theta_new, sse_new, var_error[0][0], covariance_theta_hat, sse_list


def lm_confidence_interval(theta, cov, na, nb, round_off=4):
    print("\nConfidence Interval for the estimated parameter(s)")
    lower_bound = []
    upper_bound = []
    for i in range(len(theta)):
        lower_bound.append(theta[i] - 2 * np.sqrt(cov[i, i]))
        upper_bound.append(theta[i] + 2 * np.sqrt(cov[i, i]))
    lower_bound = np.round(lower_bound, decimals=round_off)
    upper_bound = np.round(upper_bound, decimals=round_off)

    coeff_df = pd.DataFrame(columns=[' ', 'Lower Bound', 'Upper Bound'])

    for i in range(na + nb):
        if i < na:
            coeff_df.loc[len(coeff_df)] = {
                ' ': f"AR coefficient {i + 1}",
                'Lower Bound': lower_bound[i][0],
                'Upper Bound': upper_bound[i][0]
            }
        else:
            coeff_df.loc[len(coeff_df)] = {
                ' ': f"MA coefficient {i + 1}",
                'Lower Bound': lower_bound[i][0],
                'Upper Bound': upper_bound[i][0]
            }

    print_tab(coeff_df)


def lm_find_roots(theta, na, round_off=4):
    den = theta[:na]
    num = theta[na:]
    if len(den) > len(num):
        for x in range(len(den) - len(num)):
            num = np.append(num, 0)
    elif len(num) > len(den):
        for x in range(len(num) - len(den)):
            den = np.append(den, 0)
    else:
        pass
    den = np.insert(den, 0, 1)
    num = np.insert(num, 0, 1)
    print("\nRoots of numerator:", np.round(np.roots(num), decimals=round_off))
    print("Roots of denominator:", np.round(np.roots(den), decimals=round_off))


def plot_sse(sse_list, model_name):
    plt.plot(sse_list)
    plt.xlabel('Iterations')
    plt.ylabel('SSE')
    plt.title(f'SSE Learning Rate {model_name}')
    plt.xticks(np.arange(0, len(sse_list), step=1))
    plt.show()


def run_lm():
    round_off = 3
    np.random.seed(6313)
    N, na, nb, den, num, mean_e, var_e, lags = get_arma_input()
    e, y = dlsim_method(N, num, den, mean=mean_e, std=np.sqrt(var_e))
    theta, sse, var_error, covariance_theta_hat, sse_list = lm_step3(y, na, nb)
    # Q1
    theta2 = np.array(theta).reshape(-1)
    for i in range(na+nb):
        if i < na:
            print('The AR coefficient {} is: {:.3f}'.format(i + 1, np.round(theta2[i], 3)))
        else:
            print('The MA coefficient {} is: {:.3f}'.format(i + 1 - na, np.round(theta2[i], 3)))
    # Q2
    lm_confidence_interval(theta, covariance_theta_hat, na, nb, round_off=round_off)
    # Q3
    print(f"\nEstimated Covariance Matrix of estimated parameters: \n{np.round(covariance_theta_hat, decimals=round_off)}")
    # Q4
    print(f"Estimated variance of error: {round(var_error, round_off)}")
    # Q5
    lm_find_roots(theta, na, round_off=round_off)
    # Q6
    plot_sse(sse_list, f"ARMA({na},{nb})")
    print('\nUsing stats model package:')
    arma_process = sm.tsa.ArmaProcess(den, num)
    mean_y = mean_e * (1 + np.sum(den)) / (1 + np.sum(num))
    y = arma_process.generate_sample(N, scale=np.sqrt(var_e)) + mean_y
    model = sm.tsa.ARIMA(y, order=(na, 0, nb))
    results = model.fit()
    for i in range(na):
        print('The AR coefficient {} is: {:.3f}'.format(i+1, round(-results.arparams[i], 3)))
    for i in range(nb):
        print('The MA coefficient {} is: {:.3f}'.format(i+1, round(results.maparams[i], 3)))
    print(results.summary())


def check_stationarity(y):
    stationary = adf_cal(y)
    rm, rv = cal_rolling_mean_var(y)
    plot_rol_mean_var(rm, rv)
    return stationary


def run_intial_sarima():
    np.random.seed(6313)
    n, na, nb, ar, ma, mean_e, var_e, lags = get_arma_input()
    e, ts = dlsim_method(n, ma, ar, mean=mean_e, std=np.sqrt(var_e))
    plot_acf_pcf(ts, lags=20)
    stationary_1 = check_stationarity(ts)
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    stationary_2 = arma_process.isstationary
    stationary = stationary_1 and stationary_2
    return ts, ar, ma, lags, stationary


def run_gpac(ar, ma, lags, size=10):
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    ry = arma_process.acf(lags=lags)
    gpac(ry, j_max=size, k_max=size, round_off=2)


def plot_ts(data, diff=0, seas=0):
    plt.plot(data[:500], label='Original')
    if diff:
        plt.plot(diff[seas:500], label='Differenced')
    plt.xlabel('Samples')
    plt.ylabel('ARIMA Process')
    plt.title('Time Series Plot')
    plt.legend()
    plt.tight_layout()
    plt.show()


def cal_error_stat(residual_error, forecast_error, order=1, nm='Model', rnd=4):
    re = []
    lags = 20

    for lag in range(1, lags + 1):
        re.append(cal_autocorr(residual_error, lag))

    plot_acf_pcf(residual_error, lags=lags)

    Q = sm.stats.acorr_ljungbox(residual_error, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]

    DOF = lags - order
    alfa = 0.01
    chi_critical = stats.chi2.ppf(1 - alfa, DOF)

    error_stat = {
        "Model": nm,
        "Q-Value": round(Q, rnd),
        'Critical-Value': chi_critical,
        "White-Residual": 'Yes' if Q < chi_critical else 'No',
        'm_res': np.round(np.mean(residual_error), rnd),
        'mse_res': np.round(np.mean(residual_error ** 2), rnd),
        'var_res': np.round(np.var(residual_error), rnd),
        'm_pred': np.round(np.mean(forecast_error), rnd),
        'mse_pred': np.round(np.mean(forecast_error ** 2), rnd),
        'var_pred': np.round(np.var(forecast_error), rnd)
    }

    temp = pd.DataFrame(columns=range(10))
    temp.columns = list(error_stat.keys())
    temp.loc[len(temp)] = error_stat

    return temp, error_stat


def cal_reg_stat(model, x, y_tr, y_tr_pred, y_tst, y_tst_pred, nm='Model', rnd=4):

    print(model.summary())

    residual_error = y_tr - y_tr_pred
    forecast_error = y_tst - y_tst_pred

    re = []
    lags = 20

    for lag in range(1, lags + 1):
        re.append(cal_autocorr(residual_error, lag))

    plot_acf_pcf(residual_error, lags=lags)

    Q = sm.stats.acorr_ljungbox(residual_error, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]

    DOF = 20 - 1
    alfa = 0.01
    chi_critical = stats.chi2.ppf(1 - alfa, DOF)

    error_stat = {
        "Model": nm,
        "R2": round(model.rsquared, rnd),
        "Adjusted-R2": round(model.rsquared_adj, rnd),
        "AIC": round(model.aic, rnd),
        "BIC": round(model.bic, rnd),
        "F-test": round(model.f_pvalue, rnd),
        "Mean Cross-val": round(np.mean(
            cross_val_score(linear_model.LinearRegression(), x, y_tst, cv=5)
        ),4),
        "Q-Value": round(Q, rnd),
        'Critical-Value': chi_critical,
        "White-Residual": 'Yes' if Q < chi_critical else 'No',
        'mse_res': np.round(np.mean(residual_error ** 2), rnd),
        'var_res': np.round(np.var(residual_error), rnd),
        'mse_pred': np.round(np.mean(forecast_error ** 2), rnd),
        'var_pred': np.round(np.var(forecast_error), rnd)
    }

    temp = pd.DataFrame(columns=range(14))
    temp.columns = list(error_stat.keys())
    temp.loc[len(temp)] = error_stat

    return temp, error_stat



