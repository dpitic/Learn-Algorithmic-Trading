"""Pair Correlation Trading.
Pair trading mean reversion is based on the correlation between two instruments.
If a pair of stocks already has a high correlation, at some point, the
correlation is diminished, it will come back to the original level (correlation
mean value). If the stock with the lower price drops, we can long this stock
and short the other stock of this pair.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import coint

import algolib.data as data
import algolib.signals as signals


def main():
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)

    # Financial instrument symbols to load and correlate
    symbol_ids = ['SPY', 'AAPL', 'ADBE', 'LUV', 'MSFT', 'SKYW', 'QCOM',
                  'HPQ', 'JNPR', 'AMD', 'IBM']
    financial_data = data.load_financial_data(symbol_ids)
    print('Financial data:')
    print(financial_data)

    print('\nFinding co-integrated pairs ...')
    pvalues, pairs = \
        signals.find_cointegrated_pairs(financial_data['Adj Close'])
    print(pairs)

    # If a p-value is lower than 0.02 it means the null hypothesis is rejected.
    # It means the two series of prices corresponding to two different symbols
    # can be co-integrated. It means the two symbols will keep the same spread
    # on average. Red on the heatmap means the p-value is 1, which means the
    # null hypothesis is not rejected, therefore there is no significant
    # evidence that the pair of symbols are co-integrated.
    sns.heatmap(pvalues, xticklabels=symbol_ids,
                yticklabels=symbol_ids, cmap='RdYlGn_r',
                mask=(pvalues >= 0.98))

    # Use MSFT and JNPR to implement pair correlation trading strategy
    symbol1 = 'MSFT'
    symbol1_prices = financial_data['Adj Close'][symbol1]
    plt.figure()
    symbol1_prices.plot(grid=True, legend=True)
    symbol2 = 'JNPR'
    symbol2_prices = financial_data['Adj Close'][symbol2]
    symbol2_prices.plot(grid=True, legend=True)
    plt.title(f'{symbol1_prices.name} and {symbol2_prices.name} '
              f'Adjusted Close Prices')

    score, pvalue, _ = coint(symbol1_prices, symbol2_prices)
    print('Cointegration p-value:', pvalue)

    ratios = symbol1_prices / symbol2_prices
    plt.figure()
    plt.title(f'Ratio Between {symbol1_prices.name} and {symbol2_prices.name} '
              f'Adjusted Close Prices')
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
    plt.title(f'{symbol1_prices.name} and {symbol2_prices.name} '
              f'Trading Signals')
    symbol1_prices.plot(grid=True, label=f'{symbol1_prices.name}', legend=True)
    symbol1_buy[signals.zscore(ratios) > -1] = 0
    symbol1_sell[signals.zscore(ratios) < 1] = 0
    symbol1_buy.plot(color='g', linestyle='None', marker='^',
                     label='Buy Signal', legend=True, grid=True)
    symbol1_sell.plot(color='r', linestyle='None', marker='v',
                      label='Sell Signal', legend=True, grid=True)

    symbol2_prices.plot(grid=True, label=f'{symbol2_prices.name}', legend=True)
    symbol2_buy[signals.zscore(ratios) < 1] = 0
    symbol2_sell[signals.zscore(ratios) > -1] = 0
    symbol2_buy.plot(color='g', linestyle='None', marker='^', grid=True)
    symbol2_sell.plot(color='r', linestyle='None', marker='v', grid=True)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, symbol1_prices.min(), symbol2_prices.max()))

    # Create pair correlation trading strategy DataFrame to store information
    # relating to orders and positions, which will be used to calculate
    # performance of this pair correlation trading strategy.
    pair_correlation_trading_strategy = pd.DataFrame(index=symbol1_prices.index)
    pair_correlation_trading_strategy['symbol1_price'] = symbol1_prices
    pair_correlation_trading_strategy['symbol1_buy'] = \
        np.zeros(len(symbol1_prices))
    pair_correlation_trading_strategy['symbol1_sell'] = \
        np.zeros(len(symbol1_prices))
    pair_correlation_trading_strategy['symbol2_buy'] = \
        np.zeros(len(symbol1_prices))
    pair_correlation_trading_strategy['symbol2_sell'] = \
        np.zeros(len(symbol1_prices))

    # Limit the number of orders by reducing the position to one share, which
    # can be a long or short position. For a given symbol, when we have a long
    # position, a sell order is the only one that is allowed. When we have a
    # short position, a buy order is the only one that is allowed. When we have
    # no position, we can either go long (by buying) or go short (by selling).
    # Store the price used to send the orders. For the paired symbol, do the
    # opposite. When we sell symbol 1, we will buy symbol2, and vice versa.
    position = 0
    for i in range(len(symbol1_prices)):
        s1_price = symbol1_prices[i]
        s2_price = symbol2_prices[i]
        if not position and symbol1_buy[i] != 0:
            pair_correlation_trading_strategy['symbol1_buy'][i] = s1_price
            pair_correlation_trading_strategy['symbol2_sell'][i] = s2_price
            position = 1
        elif not position and symbol1_sell[i] != 0:
            pair_correlation_trading_strategy['symbol1_sell'][i] = s1_price
            pair_correlation_trading_strategy['symbol2_buy'][i] = s2_price
            position = -1
        elif position == -1 and (
                symbol1_sell[i] == 0 or i == len(symbol1_prices) - 1):
            pair_correlation_trading_strategy['symbol1_buy'][i] = s1_price
            pair_correlation_trading_strategy['symbol2_sell'][i] = s1_price
            position = 0
        elif position == 1 and (
                symbol1_buy[i] == 0 or i == len(symbol1_prices) - 1):
            pair_correlation_trading_strategy['symbol1_sell'][i] = s1_price
            pair_correlation_trading_strategy['symbol2_buy'][i] = s2_price
            position = 0

    # Plot pair correlation trading strategy
    plt.figure()
    plt.title('Pair Correlation Trading Strategy')
    symbol1_prices.plot(grid=True, label=f'{symbol1_prices.name}', legend=True)
    pair_correlation_trading_strategy['symbol1_buy'].plot(
        color='g', linestyle='None', marker='^', label='Buy Signal',
        legend=True, grid=True)
    pair_correlation_trading_strategy['symbol1_sell'].plot(
        color='r', linestyle='None', marker='v', label='Sell Signal',
        legend=True, grid=True)
    symbol2_prices.plot(grid=True, label=f'{symbol2_prices.name}', legend=True)
    pair_correlation_trading_strategy['symbol2_buy'].plot(
        color='g', linestyle='None', marker='^', label='Buy Signal', grid=True)
    pair_correlation_trading_strategy['symbol2_sell'].plot(
        color='r', linestyle='None', marker='v', label='Sell Signal', grid=True)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, symbol1_prices.min(), symbol2_prices.max()))

    print(f'\nPair correlation trading strategy {symbol1_buy.name} buy:')
    print(pair_correlation_trading_strategy['symbol1_buy'])

    # Calculate profit and loss of the pair correlation trading strategy
    pair_correlation_trading_strategy['symbol1_position'] = \
        pair_correlation_trading_strategy['symbol1_buy'] - \
        pair_correlation_trading_strategy['symbol1_sell']

    pair_correlation_trading_strategy['symbol2_position'] = \
        pair_correlation_trading_strategy['symbol2_buy'] - \
        pair_correlation_trading_strategy['symbol2_sell']
    # Plot pair correlation trading strategy positions
    plt.figure()
    # Symbol 1 P&L
    pair_correlation_trading_strategy['symbol1_position'].cumsum().plot()
    # Symbol 2 P&L
    pair_correlation_trading_strategy['symbol2_position'].cumsum().plot()

    # Calculate total P&L
    pair_correlation_trading_strategy['total_position'] = \
        pair_correlation_trading_strategy['symbol1_position'] + \
        pair_correlation_trading_strategy['symbol2_position']
    pair_correlation_trading_strategy['total_position'].cumsum().plot(
        grid=True)
    plt.title(f'{symbol1_prices.name} and {symbol2_prices.name} P&L')
    plt.legend()

    # Rerun the previous trading simulation
    pair_correlation_trading_strategy['symbol1_price'] = symbol1_prices
    pair_correlation_trading_strategy['symbol1_buy'] = \
        np.zeros(len(symbol1_prices))
    pair_correlation_trading_strategy['symbol1_sell'] = \
        np.zeros(len(symbol1_prices))
    pair_correlation_trading_strategy['symbol2_buy'] = \
        np.zeros(len(symbol1_prices))
    pair_correlation_trading_strategy['symbol2_sell'] = \
        np.zeros(len(symbol1_prices))
    pair_correlation_trading_strategy['delta'] = np.zeros(len(symbol1_prices))

    position = 0
    s1_shares = 1000000
    for i in range(len(symbol1_prices)):
        s1_positions = symbol1_prices[i] * s1_shares
        s2_positions = symbol2_prices[i] * int(s1_positions / symbol2_prices[i])
        print(symbol1_prices[i], symbol2_prices[i])
        delta_position = s1_positions - s2_positions
        if not position and symbol1_buy[i] != 0:
            pair_correlation_trading_strategy['symbol1_buy'][i] = s1_positions
            pair_correlation_trading_strategy['symbol2_sell'][i] = s2_positions
            pair_correlation_trading_strategy['delta'][i] = delta_position
            position = 1
        elif not position and symbol1_sell[i] != 0:
            pair_correlation_trading_strategy['symbol1_sell'][i] = s1_positions
            pair_correlation_trading_strategy['symbol2_buy'][i] = s2_positions
            pair_correlation_trading_strategy['delta'][i] = delta_position
            position = -1
        elif position == -1 and (
                symbol1_sell[i] == 0 or i == len(symbol1_prices) - 1):
            pair_correlation_trading_strategy['symbol1_buy'][i] = s1_positions
            pair_correlation_trading_strategy['symbol2_sell'][i] = s2_positions
            position = 0
        elif position == 1 and (
                symbol1_buy[i] == 0 or i == len(symbol1_prices) - 1):
            pair_correlation_trading_strategy['symbol1_sell'][i] = s1_positions
            pair_correlation_trading_strategy['symbol2_buy'][i] = s1_positions
            position = 0

    print(f'Pair correlation trading strategy {symbol1_prices.name} position:')
    pair_correlation_trading_strategy['symbol1_position'] = \
        pair_correlation_trading_strategy['symbol1_buy'] - \
        pair_correlation_trading_strategy['symbol1_sell']

    pair_correlation_trading_strategy['symbol2_position'] = \
        pair_correlation_trading_strategy['symbol2_buy'] - \
        pair_correlation_trading_strategy['symbol2_sell']
    # Plot pair correlation trading strategy positions
    plt.figure()
    pair_correlation_trading_strategy['symbol1_position'].cumsum().plot()
    pair_correlation_trading_strategy['symbol2_position'].cumsum().plot()

    # Total P&L
    pair_correlation_trading_strategy['total_position'] = \
        pair_correlation_trading_strategy['symbol1_position'] + \
        pair_correlation_trading_strategy['symbol2_position']
    pair_correlation_trading_strategy['total_position'].cumsum().plot(grid=True)
    plt.title(f'{symbol1_prices.name} and {symbol2_prices.name} Positions')
    plt.legend()

    # Plot delta positions
    plt.figure()
    pair_correlation_trading_strategy['delta'].plot(grid=True)
    plt.title('Delta Position')

    # Display plots and block
    plt.show()


if __name__ == '__main__':
    main()
