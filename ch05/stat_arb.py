"""Statistical arbitrage of foreign currencies."""
from itertools import cycle

import matplotlib.pyplot as plt
import pandas as pd

import algolib.data as data


def main():
    # Get daily currency data for 4 years between 2014-01-01 and 2018-01-01,
    # for 7 major currency pairs, and save in data directory
    trading_instrument = 'CADUSD=X'
    symbols = ['AUDUSD=X', 'GBPUSD=X', 'CADUSD=X', 'CHFUSD=X', 'EURUSD=X',
               'JPYUSD=X', 'NZDUSD=X']
    symbols_data = data.load_currency_data(symbols)

    # Visualise prices for currency to inspect relationships
    plot_currencies(symbols_data)
    plt.show()


def plot_currencies(symbols_data):
    """Plot currency pairs."""
    cycol = cycle('bgrcmky')
    price_data = pd.DataFrame()
    for symbol in symbols_data:
        multiplier = 1.0
        # Scale JPYUSD pair purely for visualisation scaling purposes only
        if symbol == 'JPYUSD=X':
            multiplier = 100.0

        label = symbol + ' Close Price'
        price_data = price_data.assign(label=pd.Series(
            symbols_data[symbol]['Close'] * multiplier,
            index=symbols_data[symbol].index))
        ax = price_data['label'].plot(color=next(cycol), lw=2.0, label=label)
    plt.xlabel('Date')
    plt.ylabel('Scaled Price')
    plt.legend()
    plt.grid()


if __name__ == "__main__":
    main()
