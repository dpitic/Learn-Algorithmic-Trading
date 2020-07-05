"""Support and Resistance Indicators"""
import matplotlib.pyplot as plt
import pandas as pd

from algolib.data import get_google_data
from algolib.signals import trading_support_resistance


def main():
    # Load Google finance data from Yahoo from 2014-01-01 to 2018-01-01
    goog_data_raw = get_google_data()
    # To avoid complications with stock split, we only take dates without
    # splits. Therefore only keep 620 days.
    goog_data = goog_data_raw.tail(620)
    lows = goog_data['Low']
    highs = goog_data['High']
    # Plot the data
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Google price in $')
    highs.plot(ax=ax1, color='c', lw=2.0)
    lows.plot(ax=ax1, color='y', lw=2.0)
    # Plot resistance limit
    plt.hlines(highs.head(200).max(), lows.index.values[0],
               lows.index.values[-1], linewidth=2, color='g',
               label='Resistance')
    # Plot support limit
    plt.hlines(lows.head(200).min(), lows.index.values[0],
               lows.index.values[-1], linewidth=2, color='r', label='Support')
    plt.axvline(x=lows.index.values[200],
                linewidth=2, color='b', linestyle=':')
    plt.grid()
    plt.legend()

    # Support and Resistance Trading Strategy
    goog_data = goog_data_raw
    goog_data_signal = pd.DataFrame(index=goog_data.index)
    goog_data_signal['price'] = goog_data['Adj Close']
    trading_support_resistance(goog_data_signal)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Google price in $')
    # Plot support, resistance and price
    goog_data_signal['sup'].plot(ax=ax1, color='g', lw=2.0)
    goog_data_signal['res'].plot(ax=ax1, color='b', lw=2.0)
    goog_data_signal['price'].plot(ax=ax1, color='r', lw=2.0)
    ax1.plot(goog_data_signal.loc[goog_data_signal.positions == 1.0].index,
             goog_data_signal.price[goog_data_signal.positions == 1.0],
             '^', markersize=7, color='k', label='buy')
    ax1.plot(goog_data_signal.loc[goog_data_signal.positions == -1.0].index,
             goog_data_signal.price[goog_data_signal.positions == -1.0],
             'v', markersize=7, color='k', label='sell')
    plt.legend()
    plt.grid()

    # Display plot and block
    plt.show()


if __name__ == "__main__":
    main()
