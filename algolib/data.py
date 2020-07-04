"""Data utilities"""
import pandas as pd
from pandas_datareader import data


def get_google_data(data_file, start_date, end_date):
    """Return the Google financial data from Yahoo Finance.

    This function tries to load the specified data file otherwise it downloads
    the Google financial data from Yahoo Finance.
    :param str data_file: Filename with path for the Google finance data.
    :param str start_date: Data start date (YYYY-MM-DD).
    :param str end_date: Data end date (YYYY-MM-DD).
    :return: DataFrame with Google financial data.
    """
    try:
        google_data = pd.read_pickle(data_file)
    except FileNotFoundError:
        google_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)
        google_data.to_pickle(data_file)
    return google_data
