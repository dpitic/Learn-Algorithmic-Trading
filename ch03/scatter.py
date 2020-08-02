"""Scatter matrix plot.
This module produces a scatter matrix plot of the regression trading
feature and target variables. It uses two features: open - close price and
high - low price to. The target variable is the difference in daily close price.
"""
import matplotlib.pyplot as plt
import pandas as pd

import algolib.finml as finml
from algolib.data import get_google_data


def main():
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)
    goog_data = get_google_data('data/goog_data_large.pkl',
                                start_date='2001-01-01',
                                end_date='2018-01-01')
    # Features are open - close price and high - low price
    # Target is difference in daily close price
    goog_data, x, y = finml.create_regression_trading_condition(goog_data)
    # Visualise the data
    pd.plotting.scatter_matrix(goog_data[['Open-Close', 'High-Low', 'Target']],
                               grid=True, diagonal='kde', alpha=0.5)

    # Display plots and block
    plt.show()


if __name__ == '__main__':
    main()
