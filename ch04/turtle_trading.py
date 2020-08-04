"""Turtle Trading Strategy.
This is a more advanced trading strategy where a long (buy) signal is created
when the price reaches the highest price for the last number of days specified
by the window size.
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
    # Turtle trading strategy
    ts = signals.turtle_trading(goog_data, 50)
    print('Turtle trading strategy:')
    print(ts)

    # Plot the adjusted close price
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Google price in $')
    goog_data['Adj Close'].plot(ax=ax1, color='g', lw=0.5, label='Price')

    ts['high'].plot(ax=ax1, color='g', lw=0.5, label='Highs')
    ts['low'].plot(ax=ax1, color='r', lw=0.5, label='Lows')
    ts['avg'].plot(ax=ax1, color='b', lw=0.5, label='Average')

    # Plot buy signal
    ax1.plot(ts.loc[ts.orders == 1.0].index,
             goog_data['Adj Close'][ts.orders == 1.0],
             '^', markersize=7, color='k', label='Buy')
    # Plot sell signal
    ax1.plot(ts.loc[ts.orders == -1.0].index,
             goog_data['Adj Close'][ts.orders == -1.0],
             'v', markersize=7, color='k', label='Sell')

    plt.legend()
    plt.title('Turtle Trading Strategy')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
