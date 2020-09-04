"""Utilities for Chapter 5 trading strategies."""
import statistics as stats

import pandas as pd
from matplotlib import pyplot as plt

import algolib.signals as signals


def visualise(df, apo_value_for_buy_entry, apo_value_for_sell_entry,
              num_shares_per_trade):
    """Plot trading strategy

    This is a utility function that plots the trading strategy. It produces the
    following plots:
    1. Close price, fast EMA, slow EMA.
    2. Absolute Price Oscillator.
    3. Positions.
    4. Profit and Loss.

    :param DataFrame df: Trading strategy results.
    :param int apo_value_for_buy_entry: APO trading signal value to enter
        buy orders/long positions.
    :param int apo_value_for_sell_entry: APO trading signal to enter
        sell orders/short positions.
    :param int num_shares_per_trade: Number of shares to buy/sell on every
        trade.
    """
    plt.figure()
    df['ClosePrice'].plot(color='blue', lw=3.0, legend=True, grid=True)
    df['FastEMA'].plot(color='c', lw=1.0, legend=True, grid=True)
    df['SlowEMA'].plot(color='m', lw=1.0, legend=True, grid=True)
    plt.plot(df.loc[df.Trades == 1].index,
             df.ClosePrice[df.Trades == 1],
             color='r', lw=0, marker='^', markersize=7, label='buy')
    plt.plot(df.loc[df.Trades == -1].index,
             df.ClosePrice[df.Trades == -1],
             color='g', lw=0, marker='v', markersize=7, label='sell')
    plt.legend()

    plt.figure()
    df['APO'].plot(color='k', lw=3.0, legend=True, grid=True)
    plt.plot(df.loc[df.Trades == 1].index,
             df.APO[df.Trades == 1],
             color='r', lw=0, marker='^', markersize=7, label='buy')
    plt.plot(df.loc[df.Trades == -1].index,
             df.APO[df.Trades == -1],
             color='g', lw=0, marker='v', markersize=7, label='sell')
    plt.axhline(y=0, lw=0.5, color='k')
    for i in range(apo_value_for_buy_entry, apo_value_for_sell_entry * 5,
                   apo_value_for_buy_entry):
        plt.axhline(y=i, lw=0.5, color='r')
    for i in range(apo_value_for_sell_entry, apo_value_for_sell_entry * 5,
                   apo_value_for_sell_entry):
        plt.axhline(y=i, lw=0.5, color='g')
    plt.title('Absolute Price Oscillator')
    plt.legend()

    plt.figure()
    df['Position'].plot(color='k', lw=1.0, legend=True, grid=True)
    plt.plot(df.loc[df.Position == 0].index,
             df.Position[df.Position == 0],
             color='k', lw=0, marker='.', label='flat')
    plt.plot(df.loc[df.Position > 0].index,
             df.Position[df.Position > 0],
             color='r', lw=0, marker='+', label='long')
    plt.plot(df[df.Position < 0].index,
             df.Position[df.Position < 0],
             color='g', lw=0, marker='_', label='short')
    plt.axhline(y=0, lw=0.5, color='k')
    for i in range(num_shares_per_trade, num_shares_per_trade * 25,
                   num_shares_per_trade * 5):
        plt.axhline(y=i, lw=0.5, color='r')
    for i in range(-num_shares_per_trade, -num_shares_per_trade * 25,
                   -num_shares_per_trade * 5):
        plt.axhline(y=i, lw=0.5, color='g')
    plt.title('Position')
    plt.legend()

    plt.figure()
    df['PnL'].plot(color='k', lw=1.0, legend=True, grid=True)
    plt.plot(df.loc[df.PnL > 0].index,
             df.PnL[df.PnL > 0],
             color='g', lw=0, marker='.')
    plt.plot(df.loc[df.PnL < 0].index,
             df.PnL[df.PnL < 0],
             color='r', lw=0, marker='.')
    plt.title('Profit and Loss')
    plt.legend()


def avg_sma_std_dev(prices, tail=620, sma_time_periods=20):
    """Return average standard deviation using SMA.

    This function calculates the standard deviation of tail number of prices
    using the simple moving average over the specified number of time periods.
    It also plots the prices and the standard deviation of the prices based on
    the simple moving average calculation.

    :param Series prices: Price series.
    :param tail: Tail number of prices to use, default=620.
    :param int sma_time_periods: Number of time periods to use in SMA
        calculation used to calculate the standard deviation of prices,
        default=20.
    :return: Average standard deviation using SMA.
    """
    tail_close = prices.tail(tail)
    std_dev_list = signals.standard_deviation(tail_close, sma_time_periods)
    tail_close_df = pd.DataFrame(tail_close)
    tail_close_df = tail_close_df.assign(
        std_dev=pd.Series(std_dev_list, index=tail_close.index))
    print(f'Last {tail} close prices and standard deviation of '
          f'{sma_time_periods} days SMA:')
    print(tail_close_df)
    print('\nStatistical summary:')
    print(tail_close_df.describe())
    # Average standard deviation of prices SMA over look back period
    avg_std_dev = stats.mean(std_dev_list)
    # Extract data to plot
    close_price = tail_close
    std_dev = tail_close_df['std_dev']
    # Plot last tail number prices and standard deviation
    fig = plt.figure()
    ax1 = fig.add_subplot(211, ylabel='Google price in $')
    close_price.plot(ax=ax1, color='g', lw=2.0, legend=True, grid=True)
    ax2 = fig.add_subplot(212, ylabel='Standard Deviation in $')
    std_dev.plot(ax=ax2, color='b', lw=2.0, legend=True, grid=True)
    # Plot average standard deviation of SMA
    ax2.axhline(y=avg_std_dev, color='k')
    return avg_std_dev
