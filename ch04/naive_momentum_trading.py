"""Naive Momentum Trading Strategy.
This module implements a naive momentum based trading strategy that counts the
number of times a price is improved. If the number is equal to a given threshold
it indicates a buy signal, assuming the price will keep rising. A sell signal
assumes the price will keep dropping.
"""
import matplotlib.pyplot as plt
import pandas as pd

import algolib.data as data
import algolib.signals as signals


def main():
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)
    goog_data = data.get_google_data('data/goog_data.pkl',
                                     start_date='2001-01-01',
                                     end_date='2018-01-01')
    # Naive momentum based trading strategy
    ts = signals.naive_momentum_trading(goog_data, 5)
    print('Naive momentum based trading strategy:')
    print(ts)

    # Plot the adjusted close price
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Google price in $')
    goog_data['Adj Close'].plot(ax=ax1, color='g', lw=0.5, label='Price')

    # Plot buy order
    ax1.plot(ts.loc[ts.orders == 1.0].index,
             goog_data['Adj Close'][ts.orders == 1],
             '^', markersize=7, color='k', label='Buy')

    # Plot sell order
    ax1.plot(ts.loc[ts.orders == -1.0].index,
             goog_data['Adj Close'][ts.orders == -1],
             'v', markersize=7, color='k', label='Sell')

    plt.legend()
    plt.title('Naive Momentum Trading Strategy')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
