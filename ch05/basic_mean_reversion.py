"""Mean reversion strategy using APO trading signal.

This module implements a mean reversion trading strategy that relies on the
Absolute Price Oscillator (APO) trading signal. It uses a static constant of 10
days for the fast EMA and a static constant of 40 days for the slow EMA. It
will perform buy trades when the APO signal value drops below -10 and perform
sell trades when the APO signal value goes above +10. It will check that new
trades are made at prices that are different from the last trade price to
prevent over trading. Positions are closed when the APO signal value changes
sign: close short positions when the APO goes negative and close long positions
when the APO goes positive. Positions are also closed if current open positions
are profitable above a certain amount, regardless of the APO values. This is
used to algorithmically lock profits and initiate more positions instead of
relying on the trading signal value.
"""
import matplotlib.pyplot as plt
import pandas as pd

import algolib.data as data
import algolib.signals as signals


def main():
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)
    # Get daily trading data for 4 years
    SYMBOL = 'GOOG'
    goog_data = data.get_google_data(f'data/{SYMBOL}_data.pkl',
                                     start_date='2014-01-01',
                                     end_date='2018-01-01')

    # Use close price for this analysis
    close = goog_data.loc[:, 'Close']

    # Variables for trading strategy trade, position and p&l management

    # Track buy/sell orders: buy=+1, sell=-1, no action=0
    orders = []
    # Track positions: long=+ve, short=-ve, flat/no position=0
    positions = []
    # Track total p&l
    pnls = []

    # Price at which last buy trade was made; used to prevent over trading
    last_buy_price = 0
    # Price at which last sell trade was made; used to prevent over trading
    last_sell_price = 0
    # Current position of the trading strategy
    position = 0
    # Sum of buy_trade_price and buy_trade_qty for every buy trade made since
    # last time being flat
    buy_sum_price_qty = 0
    # Summation of buy_trade_qty for every buy trade made since last time being
    # flat
    buy_sum_qty = 0
    # Sum of products of sell_trade_price and sell_trade_qty for every sell
    # trade made since last time being flat
    sell_sum_price_qty = 0
    # Sum of sell_trade_qty for every sell Trade made since last time being
    # flat
    sell_sum_qty = 0
    # Open/unrealised PnL marked to market
    open_pnl = 0
    # Closed/realised PnL so far
    closed_pnl = 0

    # Constants defining strategy behaviour/thresholds

    # APO trading signal value below which to enter buy orders/long position
    APO_VALUE_FOR_BUY_ENTRY = -10
    # APO trading signal value above which to enter sell orders/short position
    APO_VALUE_FOR_SELL_ENTRY = 10
    # Minimum price change since last trade before considering trading again.
    # This is to prevent over trading at around same prices
    MIN_PRICE_MOVE_FROM_LAST_TRADE = 10
    # Number of shares to buy/sell on every trade
    NUM_SHARES_PER_TRADE = 10
    # Minimum open/unrealised profit at which to close and lock profits
    MIN_PROFIT_TO_CLOSE = 10 * NUM_SHARES_PER_TRADE

    # Trading strategy

    # Calculate fast and slow EAM and APO on close price
    apo_df = signals.absolute_price_oscillator(close, time_period_fast=10,
                                               time_period_slow=40)
    ema_fast_values = apo_df.loc[:, 'ema_fast'].tolist()
    ema_slow_values = apo_df.loc[:, 'ema_slow'].tolist()
    apo_values = apo_df.loc[:, 'apo'].tolist()

    # Trading strategy main loop
    for close_price, apo in zip(close, apo_values):
        # Check trading signal against trading parameters/thresholds and
        # positions to trade

        # Perform a sell trade at close_price on the following conditions:
        # 1. APO trading signal value is above sell entry threshold and the
        #    difference between last trade price and current price is different
        #    enough.
        # 2. We are long (+ve position) and either APO trading signal value is
        #    at or above 0 or current position is profitable enough to lock
        #    profit.

        if ((apo > APO_VALUE_FOR_SELL_ENTRY and abs(
                close_price - last_sell_price) > MIN_PRICE_MOVE_FROM_LAST_TRADE)
                or
                (position > 0 and (
                        apo >= 0 or open_pnl > MIN_PROFIT_TO_CLOSE))):
            orders.append(-1)  # mark the sell trade
            last_sell_price = close_price
            position -= NUM_SHARES_PER_TRADE  # reduce position by size of trade
            sell_sum_price_qty += close_price * NUM_SHARES_PER_TRADE
            sell_sum_qty += NUM_SHARES_PER_TRADE
            print('Sell ', NUM_SHARES_PER_TRADE, ' @ ', close_price,
                  'Position: ', position)

        # Perform a buy trade at close_price on the following conditions:
        # 1. APO trading signal value is below buy entry threshold and the
        #    difference between last trade price and current price is different
        #    enough.
        # 2. We are short (-ve position) and either APO trading signal value is
        #    at or below 0 or current position is profitable enough to lock
        #    profit.
        elif ((apo < APO_VALUE_FOR_BUY_ENTRY and abs(
                close_price - last_buy_price) > MIN_PRICE_MOVE_FROM_LAST_TRADE)
              or
              (position < 0 and (apo <= 0 or open_pnl > MIN_PROFIT_TO_CLOSE))):
            orders.append(+1)  # mark the buy trade
            last_buy_price = close_price
            position += NUM_SHARES_PER_TRADE  # increase position by trade size
            buy_sum_price_qty += close_price * NUM_SHARES_PER_TRADE
            buy_sum_qty += NUM_SHARES_PER_TRADE
            print('Buy ', NUM_SHARES_PER_TRADE, ' @ ', close_price,
                  'Position: ', position)
        else:
            # No trade since none of the conditions were met to buy or sell
            orders.append(0)

        positions.append(position)

        # Update open/unrealised and closed/realised positions
        open_pnl = 0
        if position > 0:
            if sell_sum_qty > 0:
                # Long position and some sell trades have been made against it,
                # close that amount based on how much was sold against this
                # long position.
                open_pnl = abs(sell_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # would be if we closed at current price.
            open_pnl += abs(sell_sum_qty - position) * (
                    close_price - buy_sum_price_qty / buy_sum_qty)
        elif position < 0:
            if buy_sum_qty > 0:
                # Short position and some buy trades have been made against it,
                # close that amount based on how much was bought against this
                # short position.
                open_pnl = abs(buy_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # wold be if we closed at current price
            open_pnl += abs(buy_sum_qty - position) * (
                    sell_sum_price_qty / sell_sum_qty - close_price)
        else:
            # Flat, so update closed pnl and reset tracking variables for
            # positions and pnls
            closed_pnl += sell_sum_price_qty - buy_sum_price_qty
            buy_sum_price_qty = 0
            buy_sum_qty = 0
            sell_sum_price_qty = 0
            sell_sum_qty = 0
            last_buy_price = 0
            last_sell_price = 0

        print('OpenPnL: ', open_pnl, ' ClosedPnL: ', closed_pnl,
              ' TotalPnL', (open_pnl + closed_pnl))
        pnls.append(closed_pnl + open_pnl)

    # Prepare DataFrame from the trading strategy results
    goog_data = goog_data.assign(
        ClosePrice=pd.Series(close, index=goog_data.index))
    goog_data = goog_data.assign(
        FastEMA=pd.Series(ema_fast_values, index=goog_data.index))
    goog_data = goog_data.assign(
        SlowEMA=pd.Series(ema_slow_values, index=goog_data.index))
    goog_data = goog_data.assign(
        APO=pd.Series(apo_values, index=goog_data.index))
    goog_data = goog_data.assign(
        Trades=pd.Series(orders, index=goog_data.index))
    goog_data = goog_data.assign(
        Position=pd.Series(positions, index=goog_data.index))
    goog_data = goog_data.assign(PnL=pd.Series(pnls, index=goog_data.index))

    # Visualise
    plt.figure()
    goog_data['ClosePrice'].plot(color='blue', lw=3.0, legend=True, grid=True)
    goog_data['FastEMA'].plot(color='c', lw=1.0, legend=True, grid=True)
    goog_data['SlowEMA'].plot(color='m', lw=1.0, legend=True, grid=True)
    plt.plot(goog_data.loc[goog_data.Trades == 1].index,
             goog_data.ClosePrice[goog_data.Trades == 1],
             color='r', lw=0, marker='^', markersize=7, label='buy')
    plt.plot(goog_data.loc[goog_data.Trades == -1].index,
             goog_data.ClosePrice[goog_data.Trades == -1],
             color='g', lw=0, marker='v', markersize=7, label='sell')
    plt.legend()

    plt.figure()
    goog_data['APO'].plot(color='k', lw=3.0, legend=True, grid=True)
    plt.plot(goog_data.loc[goog_data.Trades == 1].index,
             goog_data.APO[goog_data.Trades == 1],
             color='r', lw=0, marker='^', markersize=7, label='buy')
    plt.plot(goog_data.loc[goog_data.Trades == -1].index,
             goog_data.APO[goog_data.Trades == -1],
             color='g', lw=0, marker='v', markersize=7, label='sell')
    plt.axhline(y=0, lw=0.5, color='k')
    for i in range(APO_VALUE_FOR_BUY_ENTRY, APO_VALUE_FOR_SELL_ENTRY * 5,
                   APO_VALUE_FOR_BUY_ENTRY):
        plt.axhline(y=i, lw=0.5, color='r')
    for i in range(APO_VALUE_FOR_SELL_ENTRY, APO_VALUE_FOR_SELL_ENTRY * 5,
                   APO_VALUE_FOR_SELL_ENTRY):
        plt.axhline(y=i, lw=0.5, color='g')
    plt.title('Absolute Price Oscillator')
    plt.legend()

    plt.figure()
    goog_data['Position'].plot(color='k', lw=1.0, legend=True, grid=True)
    plt.plot(goog_data.loc[goog_data.Position == 0].index,
             goog_data.Position[goog_data.Position == 0],
             color='k', lw=0, marker='.', label='flat')
    plt.plot(goog_data.loc[goog_data.Position > 0].index,
             goog_data.Position[goog_data.Position > 0],
             color='r', lw=0, marker='+', label='long')
    plt.plot(goog_data[goog_data.Position < 0].index,
             goog_data.Position[goog_data.Position < 0],
             color='g', lw=0, marker='_', label='short')
    plt.axhline(y=0, lw=0.5, color='k')
    for i in range(NUM_SHARES_PER_TRADE, NUM_SHARES_PER_TRADE * 25,
                   NUM_SHARES_PER_TRADE * 5):
        plt.axhline(y=i, lw=0.5, color='r')
    for i in range(-NUM_SHARES_PER_TRADE, -NUM_SHARES_PER_TRADE * 25,
                   -NUM_SHARES_PER_TRADE * 5):
        plt.axhline(y=i, lw=0.5, color='g')
    plt.title('Position')
    plt.legend()

    plt.figure()
    goog_data['PnL'].plot(color='k', lw=1.0, legend=True, grid=True)
    plt.plot(goog_data.loc[goog_data.PnL > 0].index,
             goog_data.PnL[goog_data.PnL > 0],
             color='g', lw=0, marker='.')
    plt.plot(goog_data.loc[goog_data.PnL < 0].index,
             goog_data.PnL[goog_data.PnL < 0],
             color='r', lw=0, marker='.')
    plt.title('Profit and Loss')
    plt.legend()

    # Save results to CSV
    goog_data.to_csv('ch05/basic_mean_reversion.csv')

    # Display plots and block
    plt.show()


if __name__ == '__main__':
    main()
