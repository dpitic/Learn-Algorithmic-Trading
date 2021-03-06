"""Dual Moving Average"""
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
    # Dual moving average trading signal
    ts = signals.dual_moving_average(goog_data, 20, 100)
    print('Dual moving average trading signal:')
    print(ts)

    # Plot curve representing orders for dual moving average strategy using
    # close price.
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Google price in $')
    goog_data['Adj Close'].plot(ax=ax1, color='g', lw=0.5, label='Price')
    ts['short_mavg'].plot(ax=ax1, color='r', lw=2.0, label='Short mavg')
    ts['long_mavg'].plot(ax=ax1, color='b', lw=2.0, label='Long mavg')

    # Plot buy order
    ax1.plot(ts.loc[ts.orders == 1.0].index,
             goog_data['Adj Close'][ts.orders == 1.0],
             '^', markersize=7, color='k', label='Buy')

    # Plot sell order
    ax1.plot(ts.loc[ts.orders == -1.0].index,
             goog_data['Adj Close'][ts.orders == -1.0],
             'v', markersize=7, color='k', label='Sell')

    plt.legend()
    plt.title('Dual Moving Average Trading Strategy')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
