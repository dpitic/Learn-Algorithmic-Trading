"""Support and Resistance Indicators"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algolib.data import get_google_data


def trading_support_resistance(data, bin_width=20):
    """Support and Resistance Trading Strategy

    A buy order is sent when a price stays in the resistance tolerance margin
    for 2 consecutive days, and a sell order when a price stays in the support
    tolerance margin for 2 consecutive days.

    :param DataFrame data: data signal.
    :param int bin_width: Number of days for rolling average.
    """
    data['sup_tolerance'] = pd.Series(np.zeros(len(data)))
    data['res_tolerance'] = pd.Series(np.zeros(len(data)))
    data['sup_count'] = pd.Series(np.zeros(len(data)))
    data['res_count'] = pd.Series(np.zeros(len(data)))
    data['sup'] = pd.Series(np.zeros(len(data)))
    data['res'] = pd.Series(np.zeros(len(data)))
    data['positions'] = pd.Series(np.zeros(len(data)))
    data['signal'] = pd.Series(np.zeros(len(data)))
    in_support = 0
    in_resistance = 0

    for x in range((bin_width - 1) + bin_width, len(data)):
        data_section = data[x - bin_width:x + 1]
        support_level = min(data_section['price'])
        resistance_level = max(data_section['price'])
        range_level = resistance_level - support_level
        data['res'][x] = resistance_level
        data['sup'][x] = support_level
        data['sup_tolerance'][x] = support_level + 0.2 * range_level
        data['res_tolerance'][x] = resistance_level - 0.2 * range_level

        if data['res_tolerance'][x] <= data['price'][x] <= data['res'][x]:
            in_resistance += 1
            data['res_count'][x] = in_resistance
        elif data['sup_tolerance'][x] >= data['price'][x] >= data['sup'][x]:
            in_support += 1
            data['sup_count'][x] = in_support
        else:
            in_support = 0
            in_resistance = 0

        if in_resistance > 2:
            data['signal'][x] = 1
        elif in_support > 2:
            data['signal'][x] = 0
        else:
            data['signal'][x] = data['signal'][x - 1]

    data['positions'] = data['signal'].diff()


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
