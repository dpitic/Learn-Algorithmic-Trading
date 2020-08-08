"""Pairs Correlation Trading Demonstration.
This module creates an artificial pair of symbols to demonstrate the concept of
pair correlation trading.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

import algolib.signals as signals


def main():
    # Seed random number generator to make results reproducible
    np.random.seed(123)

    # Symbol 1 daily returns, starting at $10 and varying randomly
    symbol1_returns = np.random.normal(0, 1, 100)
    # Symbol 1 series for prices
    symbol1_prices = pd.Series(np.cumsum(symbol1_returns), name='Symbol1') + 10

    # Symbol 2 daily returns based on Symbol 1 with noise for market fluctuation
    noise = np.random.normal(0, 1, 100)
    symbol2_prices = symbol1_prices + 10 + noise
    symbol2_prices.name = 'Symbol2'

    # Visualise the daily price variation for both symbols
    plt.figure()
    plt.title('Symbol 1 and Symbol 2 Prices')
    symbol1_prices.plot(grid=True, legend=True)
    symbol2_prices.plot(grid=True, legend=True)

    # Calculate cointegration between the symbols
    # If a p-value is lower than 0.02 it means the null hypothesis is rejected.
    # It means the two series of prices corresponding to two different symbols
    # can be co-integrated. It means the two symbols will keep the same spread
    # on average. Red on the heatmap means the p-value is 1, which means the
    # null hypothesis is not rejected, therefore there is no significant
    # evidence that the pair of symbols are co-integrated.
    score, pvalue, _ = coint(symbol1_prices, symbol2_prices)
    print('Cointegration p-value:', pvalue)

    # We need to set the threshold that defines when a given price is far off
    # the mean price. That will need to use specific values for a given symbol.
    # If there are many symbols then this will have to be performed for all of
    # them. To avoid tedious work, normalise this study by analysing the ratio
    # of the two prices instead.
    ratios = symbol1_prices / symbol2_prices
    plt.figure()
    plt.title('Ratios Between Symbol 1 and Symbol 2 Price')
    ratios.plot(grid=True)

    # Visualise when to place orders with z-score evolution
    plt.figure()
    signals.zscore(ratios).plot(grid=True, label='z-score')
    plt.title('Z-score Evolution')
    plt.axhline(signals.zscore(ratios).mean(), color='k')
    plt.axhline(1.0, color='r')
    plt.axhline(-1.0, color='g')

    # Each time the z-score reaches one of the thresholds we have a trading
    # signal. Going long for Symbol 1 means sending a buy order for Symbol 1
    # and a sell order for Symbol 2 concurrently.
    plt.figure()
    ratios.plot(grid=True, label='Ratio', legend=True)
    buy = ratios.copy()
    sell = ratios.copy()
    buy[signals.zscore(ratios) > -1] = 0
    sell[signals.zscore(ratios) < 1] = 0
    buy.plot(color='g', linestyle='None', marker='^', label='Buy Signal',
             legend=True, grid=True)
    sell.plot(color='r', linestyle='None', marker='v', label='Sell Signal',
              legend=True, grid=True)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, ratios.min(), ratios.max()))

    # Buy and sell orders for each symbol
    symbol1_buy = symbol1_prices.copy()
    symbol1_sell = symbol1_prices.copy()
    symbol2_buy = symbol2_prices.copy()
    symbol2_sell = symbol2_prices.copy()

    plt.figure()
    plt.title('Symbol 1 and 2 Trading Signals')
    symbol1_prices.plot(grid=True, label='Symbol 1', legend=True)
    symbol1_buy[signals.zscore(ratios) > -1] = 0
    symbol1_sell[signals.zscore(ratios) < 1] = 0
    symbol1_buy.plot(color='g', linestyle='None', marker='^',
                     label='Buy Signal', legend=True, grid=True)
    symbol1_sell.plot(color='r', linestyle='None', marker='v',
                      label='Sell Signal', legend=True, grid=True)

    symbol2_prices.plot(grid=True, label='Symbol 2', legend=True)
    symbol2_buy[signals.zscore(ratios) < 1] = 0
    symbol2_sell[signals.zscore(ratios) > -1] = 0
    symbol2_buy.plot(color='g', linestyle='None', marker='^', grid=True)
    symbol2_sell.plot(color='r', linestyle='None', marker='v', grid=True)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, symbol1_prices.min(), symbol2_prices.max()))

    # Show plots and block
    plt.show()


if __name__ == '__main__':
    main()
