"""Mean reversion strategy using volatility adjusted APO trading signal.

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
import statistics as stats

import matplotlib.pyplot as plt
import pandas as pd

import algolib.data as data
import algolib.signals as signals
import viz_utils


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

    # Preliminary analysis

    # Calculate and plot SMA to determine average SMA standard deviation
    # To avoid complications with stock split, we only take dates without
    # splits. Therefore only keep 620 days.
    tail_close = close.tail(620)
    sma_time_periods = 20  # look back period
    std_dev_list = signals.standard_deviation(tail_close, sma_time_periods)
    tail_close_df = pd.DataFrame(tail_close)
    tail_close_df = tail_close_df.assign(
        std_dev=pd.Series(std_dev_list, index=tail_close.index))
    print(f'Last 620 close prices and standard deviation of '
          f'{sma_time_periods} days SMA:')
    print(tail_close_df)
    print('\nStatistical summary:')
    print(tail_close_df.describe())

    # Average standard deviation of prices SMA over look back period
    avg_std_dev = 15
    print(f'\nAverage stdev of prices SMA over {sma_time_periods} day '
          f'look back period: {avg_std_dev}')

    # Extract data to plot
    close_price = tail_close
    std_dev = tail_close_df['std_dev']

    # Plot last 620 prices and standard deviation
    fig = plt.figure()
    ax1 = fig.add_subplot(211, ylabel='Google price in $')
    close_price.plot(ax=ax1, color='g', lw=2.0, legend=True, grid=True)
    ax2 = fig.add_subplot(212, ylabel='Standard Deviation in $')
    std_dev.plot(ax=ax2, color='b', lw=2.0, legend=True, grid=True)
    # Plot average standard deviation of SMA
    ax2.axhline(y=stats.mean(std_dev_list), color='k')

    # Constants defining strategy behaviour/thresholds

    # APO trading signal value below which to enter buy orders/long position
    apo_value_for_buy_entry = -10
    # APO trading signal value above which to enter sell orders/short position
    apo_value_for_sell_entry = 10
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
    print('\nTrading strategy:')
    df = signals.volatility_mean_reversion(
        close, sma_time_periods, avg_std_dev, ema_time_period_fast,
        ema_time_period_slow, apo_value_for_buy_entry, apo_value_for_sell_entry,
        min_price_move_from_last_trade, num_shares_per_trade,
        min_profit_to_close)

    # Visualise
    viz_utils.visualise(df, apo_value_for_buy_entry, apo_value_for_sell_entry,
                        num_shares_per_trade)

    # Prepare DataFrame to save results to CSV
    goog_data = pd.concat([goog_data, df], axis=1)
    # Remove redundant close price column
    goog_data = goog_data.drop('ClosePrice', axis=1)
    goog_data.to_csv('ch05/volatility_mean_reversion.csv')

    # Display plots and block
    plt.show()


if __name__ == '__main__':
    main()
