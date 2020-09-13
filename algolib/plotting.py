"""Utilities visualising trading strategies."""

import matplotlib.pyplot as plt


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
    plt.title('Close Price $')
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
