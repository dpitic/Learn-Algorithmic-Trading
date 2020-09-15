"""Mean reversion volatility adjusted APO trading strategy with dynamic risk.

This module implements a mean reversion trading strategy that relies on the
Absolute Price Oscillator (APO) trading signal. It incorporates risk management
strategies using dynamically adjusted risk limits between predefined limits.

It uses a static constant of 10 days for the fast EMA and a static constant of
40 days for the slow EMA. It will perform buy trades when the APO signal value
drops below -10 and perform sell trades when the APO signal value goes above
+10. It will check that new trades are made at prices that are different from
the last trade price to prevent over trading. Positions are closed when the APO
signal value changes sign: close short positions when the APO goes negative and
close long positions when the APO goes positive. Positions are also closed if
current open positions are profitable above a certain amount, regardless of the
APO values. This is used to algorithmically lock profits and initiate more
positions instead of relying on the trading signal value.
"""

import matplotlib.pyplot as plt
import pandas as pd

import algolib.data as data
import algolib.plotting as plotting
import algolib.signals as signals

# Set risk limits to 150% of the maximum achieved historically
RISK_BUFFER_FACTOR = 1.5


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
    sma_time_periods = 20  # look back period
    avg_std_dev = signals.avg_sma_std_dev(close, tail=620,
                                          sma_time_periods=sma_time_periods)
    print(f'\nAverage stdev of prices SMA over {sma_time_periods} day '
          f'look back period: {avg_std_dev}')

    # Average standard deviation of prices SMA over look back period
    avg_std_dev = 15
    print(f'\nApproximate average stdev of prices SMA over '
          f'{sma_time_periods} day look back period: {avg_std_dev}')

    # Constants defining strategy behaviour/thresholds

    # APO trading signal value below which to enter buy orders/long position
    apo_value_for_buy_entry = -10
    # APO trading signal value above which to enter sell orders/short position
    apo_value_for_sell_entry = 10
    # Minimum price change since last trade before considering trading again.
    # This is to prevent over trading at around same prices
    min_price_move_from_last_trade = 10
    # Range of number of shares to buy/sell on every trade
    min_num_shares_per_trade = 1
    max_num_shares_per_trade = 50
    increment_num_shares_per_trade = 2
    # Beginning number of shares to buy/sell on every trade
    num_shares_per_trade = min_num_shares_per_trade
    # Minimum open/unrealised profit at which to close and lock profits
    min_profit_to_close = 10 * num_shares_per_trade

    # Performance and risk limits
    risk_limit_weekly_stop_loss = -6000
    increment_risk_limit_weekly_stop_loss = -12000
    risk_limit_monthly_stop_loss = -15000
    increment_risk_limit_monthly_stop_loss = -30000
    risk_limit_max_positions = 5
    increment_risk_limit_max_positions = 3
    risk_limit_max_positions_holding_time_days = 120 * risk_limit_max_positions
    risk_limit_max_trade_size = 5
    increment_risk_limit_max_trade_size = 2
    risk_limit_max_traded_volume = 4000 * RISK_BUFFER_FACTOR

    # Exponential moving average time periods for APO calculation
    ema_time_period_fast = 10
    ema_time_period_slow = 40

    # Calculate trading strategy
    print('\nTrading strategy:')
    df = signals.volatility_mean_reversion_dynamic_risk(
        close,
        risk_limit_weekly_stop_loss,
        increment_risk_limit_weekly_stop_loss,
        risk_limit_monthly_stop_loss,
        increment_risk_limit_monthly_stop_loss,
        risk_limit_max_positions,
        increment_risk_limit_max_positions,
        risk_limit_max_positions_holding_time_days,
        risk_limit_max_trade_size,
        increment_risk_limit_max_trade_size,
        risk_limit_max_traded_volume,  # TODO: check if still required
        sma_time_periods,
        avg_std_dev,
        ema_time_period_fast,
        ema_time_period_slow,
        apo_value_for_buy_entry,
        apo_value_for_sell_entry,
        min_price_move_from_last_trade,
        min_num_shares_per_trade,
        max_num_shares_per_trade,
        increment_num_shares_per_trade,
        min_profit_to_close)

    # Visualise
    plotting.visualise(df, apo_value_for_buy_entry, apo_value_for_sell_entry,
                       num_shares_per_trade)
    # Additional plots
    plt.figure()
    plt.title('Number of Shares and Maximum Trade Size')
    df['NumShares'].plot(color='b', lw=3.0)
    df['MaxTradeSize'].plot(color='g', lw=1.0)
    plt.legend()
    plt.grid()

    plt.figure()
    plt.title('Absolute and Maximum Positions')
    df['AbsPosition'].plot(color='b', lw=1.0)
    df['MaxPosition'].plot(color='g', lw=1.0)
    plt.legend()
    plt.grid()

    # Prepare DataFrame to save results to CSV
    goog_data = pd.concat([goog_data, df], axis=1)
    # Remove redundant close price column
    goog_data = goog_data.drop('ClosePrice', axis=1)
    goog_data.to_csv('ch06/volatility_mean_reversion_dynamic_risk.csv')

    # Display plots and block
    plt.show()


if __name__ == '__main__':
    main()
