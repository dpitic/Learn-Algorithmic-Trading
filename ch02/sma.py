"""Simple Moving Average Trading Signal
This module implements a simple moving average that computes an average over a
20 day moving window and compares the SMA values against daily prices.
"""
import statistics as stats

import matplotlib.pyplot as plt
import pandas as pd

from algolib.data import get_google_data


def simple_moving_average(series, time_period=20):
    """Return Simple Moving Average (SMA)

    SMA is calculated by adding the price of an instrument over a number of time
    periods and then dividing the sum by the number of time periods. The SMA is
    basically the average price of the given time period, with equal weighting
    given to the price of each period.

    SMA = (sum(price, n)) / n

    Where: n = time period

    :param Series series: Price series.
    :param int time_period: Number of days over which to average, default=20
    :return: List of SMA prices.
    """
    history = []  # track history of prices
    sma_values = []  # track simple moving average values
    for price in series:
        history.append(price)
        # Remove oldest price because we only average over last time_period
        if len(history) > time_period:
            del history[0]
        sma_values.append(stats.mean(history))
    return sma_values


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
