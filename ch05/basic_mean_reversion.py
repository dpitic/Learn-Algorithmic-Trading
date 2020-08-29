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
    close = goog_data.loc['Close']

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
    ema_fast_values = apo_df.loc['ema_fast'].tolist()
    ema_slow_values = apo_df.loc['ema_slow'].tolist()
    apo_values = apo_df.loc['apo'].tolist()


if __name__ == '__main__':
    main()
