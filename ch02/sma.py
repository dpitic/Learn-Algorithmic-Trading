"""Simple Moving Average Trading Signal
This module implements a simple moving average that computes an average over a
20 day moving window and compares the SMA values against daily prices.
"""
import matplotlib.pyplot as plt
import pandas as pd

from algolib.data import get_google_data
from algolib.signals import simple_moving_average


def main():
    # Get Google data from Yahoo from 2014-01-01 to 2018-01-01
    goog_data_raw = get_google_data()
    goog_data = goog_data_raw.tail(620)
    # Use close price for this analysis
    close = goog_data['Close']
    close_sma = simple_moving_average(close)
    goog_data = goog_data.assign(
        ClosePrice=pd.Series(close, index=goog_data.index))
    goog_data = goog_data.assign(
        Simple20DayMovingAverage=pd.Series(close_sma, index=goog_data.index))

    close_price = goog_data['ClosePrice']
    sma = goog_data['Simple20DayMovingAverage']

    # Plot the close price 20 day SMA
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Google price in $')
    close_price.plot(ax=ax1, color='g', lw=2.0, legend=True)
    sma.plot(ax=ax1, color='r', lw=2.0, legend=True)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
