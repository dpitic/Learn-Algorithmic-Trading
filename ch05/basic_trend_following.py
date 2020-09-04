"""Trend following trading strategy using APO trading signal.

This module implements a mean reversion trading strategy that relies on the
Absolute Price Oscillator (APO) trading signal. It uses a static constant of 10
days for the fast EMA and a static constant of 40 days for the slow EMA. It
will perform buy trades when the APO signal value goes above 10 and perform
sell trades when the APO signal value goes below -10. It will check that new
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
import ch05_utils


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

    # Constants defining strategy behaviour/thresholds

    # APO trading signal value above which to enter buy orders/long position
    apo_value_for_buy_entry = 10
    # APO trading signal below above which to enter sell orders/short position
    apo_value_for_sell_entry = -10
    # Minimum price change since last trade before considering trading again.
    # This is to prevent over trading at around same prices
    min_price_move_from_last_trade = 10
    # Number of shares to buy/sell on every trade
    num_shares_per_trade = 10
    # Minimum open/unrealised profit at which to close and lock profits
    min_profit_to_close = 10 * num_shares_per_trade

    # Exponential moving average time periods for APO calculation
    ema_time_period_fast = 10
    ema_time_period_slow = 40

    # Calculate trading strategy
    df = signals.basic_trend_following(
        close, ema_time_period_fast, ema_time_period_slow,
        apo_value_for_buy_entry=apo_value_for_buy_entry,
        apo_value_for_sell_entry=apo_value_for_sell_entry,
        min_price_move_from_last_trade=min_price_move_from_last_trade,
        num_shares_per_trade=num_shares_per_trade,
        min_profit_to_close=min_profit_to_close)

    # Visualise
    ch05_utils.visualise(df, apo_value_for_buy_entry, apo_value_for_sell_entry,
                         num_shares_per_trade)

    # Prepare DataFrame to save results to CSV
    goog_data = pd.concat([goog_data, df], axis=1)
    # Remove redundant close price column
    goog_data = goog_data.drop('ClosePrice', axis=1)
    goog_data.to_csv('ch05/basic_trend_following.csv')

    # Display plots and block
    plt.show()


if __name__ == '__main__':
    main()
