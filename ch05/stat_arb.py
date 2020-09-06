"""Statistical arbitrage of foreign currencies."""
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import algolib.data as data
import algolib.signals as signals


def main():
    # Get daily currency data for 4 years between 2014-01-01 and 2018-01-01,
    # for 7 major currency pairs, and save in data directory
    trading_instrument = 'CADUSD=X'
    symbols = ['AUDUSD=X', 'GBPUSD=X', 'CADUSD=X', 'CHFUSD=X', 'EURUSD=X',
               'JPYUSD=X', 'NZDUSD=X']
    symbols_data = data.load_currency_data(symbols)

    # Visualise prices for currency to inspect relationships
    plot_currencies(symbols_data)

    # Statistical arbitration signal parameters
    sma_time_periods = 20  # SMA look back period
    # Look back period of close price deviations from SMA
    price_dev_num_prices = 200

    # Constants defining trading strategy behaviour/thresholds

    # Value above which to enter buy orders/long position
    stat_arb_value_for_buy_entry = 0.01
    # Value below which to enter sell orders/short position
    stat_arb_value_for_sell_entry = -0.01
    # Minimum price change since last trade before considering trading again;
    # this prevents over trading at/around same prices
    min_price_move_from_last_trade = 0.01
    # Number of currency to buy/sell on every trade
    num_shares_per_trade = 1000000
    # Minumum open/unrealised profit at which to close and lock profits
    min_profit_to_close = 10

    # Calculate trading strategy
    print('\nTrading strategy:')
    df = signals.currency_stat_arb(symbols_data, trading_instrument,
                                   sma_time_periods, price_dev_num_prices,
                                   stat_arb_value_for_buy_entry,
                                   stat_arb_value_for_sell_entry,
                                   min_price_move_from_last_trade,
                                   num_shares_per_trade, min_profit_to_close)

    # Visualise
    visualise(df, stat_arb_value_for_buy_entry, stat_arb_value_for_sell_entry,
              num_shares_per_trade)

    # Save statistical arbitrage trading strategy results
    df.to_csv('ch05/statistical_arbitrage.csv')

    # Display plots and block
    plt.show()


def plot_currencies(symbols_data):
    """Plot currency pairs.

    Plot the currency pairs in the dictionary. The dictionary keys are the
    currency pair name, and the values are the DataFrames containing historical
    financial data.

    :param dict symbols_data: Dictionary of foreign currency historical data.
    """
    cycol = cycle('bgrcmky')
    plt.figure()
    plt.title('Currency Pairs')
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


def visualise(df, stat_arb_value_for_buy_entry, stat_arb_value_for_sell_entry,
              num_shares_per_trade):
    plt.figure()
    plt.title('Close Price $')
    plt.plot(df.index, df.ClosePrice, color='k', lw=1.0, label='Close Price')

    plt.plot(df.loc[df.Trades == 1].index, df.ClosePrice[df.Trades == 1],
             color='r', lw=0, marker='^', markersize=7, label='buy')
    plt.plot(df.loc[df.Trades == -1].index, df.ClosePrice[df.Trades == -1],
             color='g', lw=0, marker='v', markersize=7, label='sell')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.title('Final Statistical Arbitrage Trading Signal')
    plt.plot(df.index, df.FinalStatArbTradingSignal, color='k', lw=1.0,
             label='Final Stat Arb Trading Signal')
    plt.plot(df.loc[df.Trades == 1].index,
             df.FinalStatArbTradingSignal[df.Trades == 1], color='r', lw=0,
             marker='^', markersize=7, label='buy')
    plt.plot(df.loc[df.Trades == -1].index,
             df.FinalStatArbTradingSignal[df.Trades == -1], color='g', lw=0,
             marker='v', markersize=7, label='sell')
    plt.axhline(y=0, lw=0.5, color='k')
    for i in np.arange(stat_arb_value_for_buy_entry,
                       stat_arb_value_for_buy_entry * 10,
                       stat_arb_value_for_buy_entry * 2):
        plt.axhline(y=i, lw=0.5, color='r')
    for i in np.arange(stat_arb_value_for_sell_entry,
                       stat_arb_value_for_sell_entry * 10,
                       stat_arb_value_for_sell_entry * 2):
        plt.axhline(y=i, lw=0.5, color='g')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.title('Position')
    plt.plot(df.index, df.Position, color='k', lw=1.0)
    plt.plot(df.loc[df.Position == 0].index,
             df.Position[df.Position == 0], color='k', lw=0, marker='.',
             label='flat')
    plt.plot(df.loc[df.Position > 0].index, df.Position[df.Position > 0],
             color='r', lw=0, marker='+', label='long')
    plt.plot(df.loc[df.Position < 0].index, df.Position[df.Position < 0],
             color='g', lw=0, marker='_', label='short')
    plt.axhline(y=0, lw=0.5, color='k')
    for i in range(num_shares_per_trade, num_shares_per_trade * 5,
                   num_shares_per_trade):
        plt.axhline(y=i, lw=0.5, color='r')
    for i in range(-num_shares_per_trade, -num_shares_per_trade * 5,
                   -num_shares_per_trade):
        plt.axhline(y=i, lw=0.5, color='g')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.title('Stat Arb Profit and Loss')
    plt.plot(df.index, df.PnL, color='k', lw=1.0, label='Profit & Loss')
    plt.plot(df.loc[df.PnL > 0].index, df.PnL[df.PnL > 0], color='g', lw=0,
             marker='.', label='Profit')
    plt.plot(df.loc[df.PnL < 0].index, df.PnL[df.PnL < 0], color='r', lw=0,
             marker='.', label='Loss')
    plt.legend()
    plt.grid()


if __name__ == "__main__":
    main()
