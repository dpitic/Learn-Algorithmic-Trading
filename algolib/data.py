"""Data utilities"""
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data
from statsmodels.tsa.stattools import adfuller


def get_google_data(data_file='data/goog_data.pkl', start_date='2014-01-01',
                    end_date='2018-01-01'):
    """Return the Google financial data from Yahoo Finance.

    This function tries to load the specified data file otherwise it downloads
    the Google financial data from Yahoo Finance.
    :param str data_file: Filename with path for the Google finance data,
        default='data/goog_data.pkl'.
    :param str start_date: Data start date, default='2014-01-01'.
    :param str end_date: Data end date, default='2018-01-01'.
    :return: DataFrame with Google financial data.
    """
    try:
        google_data = pd.read_pickle(data_file)
    except FileNotFoundError:
        google_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)
        google_data.to_pickle(data_file)
    return google_data


def load_financial_data(symbols, data_file='data/multi_data_large.pkl',
                        start_date='2001-01-01', end_date='2018-01-01'):
    """Return financial data for the list of symbols for Yahoo Finance.

    :param list symbols: Financial symbols to load.
    :param str data_file: Filename with path for financial data,
        default='data/multi_data_large.pkl'.
    :param str start_date: Data start date, default='2001-01-01'.
    :param str end_date: Data end date, default='2018-01-01'.
    :return: DataFrame with financial data for specified symbols.
    """
    try:
        df = pd.read_pickle(data_file)
    except FileNotFoundError:
        df = data.DataReader(symbols, 'yahoo', start_date, end_date)
        df.to_pickle(data_file)
    return df


def plot_rolling_statistics_ts(time_series, title_text, ytext, window_size=12):
    """Plot original, rolling average and rolling standard deviation.

    :param Series time_series: Time series to plot.
    :param str title_text: Plot title.
    :param str ytext: y-axis label.
    :param int window_size: Moving average window size, default=12.
    """
    plt.figure()
    time_series.plot(color='red', label='Original', lw=0.5, grid=True)
    time_series.rolling(window_size).mean().plot(
        color='blue', label='Rolling Mean', grid=True)
    time_series.rolling(window_size).std().plot(
        color='black', label='Rolling Std', grid=True)

    plt.legend(loc='best')
    plt.ylabel(ytext)
    plt.title(title_text)


def plot_stationary_ts(time_series, title_text, ytext, window_size=12):
    """Plot stationary time series, rolling average and rolling std. dev.

    :param Series time_series: Time series to plot.
    :param str title_text: Plot title.
    :param str ytext: y-axis label.
    :param int window_size: Moving average window size, default=12.
    """
    plt.figure()
    (time_series - time_series.rolling(window_size).mean()).plot(
        color='red', label='Stationary', lw=0.5, grid=True)
    (time_series - time_series.rolling(window_size).mean()).rolling(
        window_size).mean().plot(
        color='blue', label='Rolling Mean', grid=True)
    (time_series - time_series.rolling(window_size).mean()).rolling(
        window_size).std().plot(
        color='black', label='Rolling Std', grid=True)

    plt.legend()
    plt.ylabel(ytext)
    plt.title(title_text)


def augmented_dickey_fuller(time_series):
    """Return the augmented Dickey-Fuller test results on time series data.

    :param Series time_series: Time series data.
    :return: Series containing the Augmented Dickey-Fuller test results
        consisting of test statistic, p-value, number of lags used, and
        number of observations used.
    """
    adf_test = adfuller(time_series, autolag='AIC')
    adf_output = pd.Series(
        adf_test[0:4],
        index=['Test Statistic', 'p-value', '# Lags Used',
               '# Observations Used'])
    return adf_output
