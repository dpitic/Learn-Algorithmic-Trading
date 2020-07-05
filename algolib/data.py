"""Data utilities"""
import pandas as pd
from pandas_datareader import data


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
