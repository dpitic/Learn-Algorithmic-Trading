"""Trading strategy risk measures.
This module implements and explores several trading risk measures using the
market data and trading strategy results from the volatility adjusted mean
reversion trading strategy..
"""
from statistics import stdev, mean

import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)
    # Load trading strategy results and data, and display on screen
    results = pd.read_csv('ch05/volatility_mean_reversion.csv')
    print(results)

    # Stop loss
    # Stop loss or max loss risk limit is the maximum amount of money a strategy
    # is allowed to lose, i.e. the minimum PnL allowed. This often has a time
    # frame for that loss such as a day, a week, or a month, or for the entire
    # lifetime of the trading strategy. A stop loss with a time frame of a day
    # means that if the strategy loses a stop loss amount of money in a single
    # day, it is not allowed to trade any more on that day, but it can resume
    # the next day. The code below computes the stop loss levels for a week and
    # a month.
    num_days = len(results.index)
    pnl = results['PnL']
    weekly_losses = []
    monthly_losses = []

    for i in range(num_days):
        if i >= 5 and pnl[i - 5] > pnl[i]:
            weekly_losses.append(pnl[i] - pnl[i - 5])

        if i >= 20 and pnl[i - 20] > pnl[i]:
            monthly_losses.append(pnl[i] - pnl[i - 20])

    plt.figure()
    plt.hist(weekly_losses, 50)
    plt.gca().set(title='Stop Loss Weekly Loss Distribution', xlabel='$',
                  ylabel='Frequency')

    plt.figure()
    plt.hist(monthly_losses, 50)
    plt.gca().set(title='Stop Loss Monthly Loss Distribution', xlabel='$',
                  ylabel='Frequency')

    # Max drawdown
    # This is also a PnL metric, but this measures the maximum loss that a
    # strategy can accept over a series of days. This is defined as the peak to
    # trough decline in a trading strategy's account value. This is important
    # as a risk measure because it provides an idea of what the historical
    # maximum decline in the account value can be. Having an expectation of what
    # the maximum drawdown is can help us understand whether the strategy loss
    # streak is still within expectations or whether something unprecedented is
    # happening.
    max_pnl = 0
    max_drawdown = 0
    drawdown_max_pnl = 0
    drawdown_min_pnl = 0

    for i in range(num_days):
        max_pnl = max(max_pnl, pnl[i])
        drawdown = max_pnl - pnl[i]

        if drawdown > max_drawdown:
            max_drawdown = drawdown
            drawdown_max_pnl = max_pnl
            drawdown_min_pnl = pnl[i]

    print('\nMax Drawdown:', max_drawdown)

    plt.figure()
    plt.title('Max Drawdown')
    results['PnL'].plot(x='Date', legend=True)
    plt.axhline(y=drawdown_max_pnl, color='g')
    plt.axhline(y=drawdown_min_pnl, color='r')
    plt.grid()

    # Position limits
    # These are the maximum positions, long or short, that the strategy should
    # have at any point in its trading lifetime. It is possible to have two
    # different position limits, one for the maximum long position and another
    # for the maximum short position, which can be useful where shorting stocks
    # have different rules and risks associated with them than being long on
    # stocks does. Every unit of open position has a risk associated with it.
    # Generally, the larger the position a strategy puts on, the larger the risk
    # associated with it. The best strategies are the ones that can make money
    # while getting into the smallest positions possible. Before a strategy is
    # deployed to production, it is important to quantify and estimate what the
    # maximum positions the strategy can get into, based on historical
    # performance, to provide indications of when a strategy is within its
    # normal behaviour parameters and when it is outside of historical norms.
    position = results['Position']
    plt.figure()
    plt.hist(position, 20)
    plt.gca().set(title='Position Limits Position Distribution',
                  xlabel='Shares', ylabel='Frequency')

    # Position holding time
    # While analysing positions that a trading strategy gets into, it is also
    # important to measure how long a position stays open until it is closed
    # and return to its flat position or opposite position. The longer a
    # position stays open, the more risk it is taking on, because the more time
    # there is for markets to make massive moves that can potentially go against
    # the open position. A long position is initiated when the position goes
    # from being short or flat to being long and is closed when the position
    # goes back to flat or short. Similarly, short positions are initiated when
    # the position goes from being long or flat to being short and is closed
    # when the position goes back to flat or long.
    position_holding_times = []
    current_pos = 0
    current_pos_start = 0
    for i in range(num_days):
        pos = results['Position'].iloc[i]

        # Flat and starting a new position
        if current_pos == 0:
            if pos != 0:
                current_pos = pos
                current_pos_start = i
            continue

        # Going from long position to flat or short position, or going from
        # short position to flat or long position
        if current_pos * pos <= 0:
            current_pos = pos
            position_holding_times.append(i - current_pos_start)
            current_pos_start = i

    print('\nPosition holding times:')
    print(position_holding_times)
    plt.figure()
    plt.hist(position_holding_times, 100)
    plt.gca().set(title='Position Holding Time Distribution',
                  xlabel='Holding time (days)', ylabel='Frequency')

    # Variance of PnLs
    # It is important to measure how much the PnLs can vary from day to day or
    # even from week to week as a measure of risk because if a trading strategy
    # has large swings in PnLs, the account value is very volatile and it is
    # hard to run a trading strategy with such a profile. Normally the standard
    # deviation of returns is computed over different days or weeks, or
    # whatever timeframe is chosen to use as the investment time horizon. Most
    # optimisation methods try to find optimal trading performance as a balance
    # between PnLs and the standard deviation of returns. The following code
    # computes the standard deviation of weekly returns.
    last_week = 0
    weekly_pnls = []
    weekly_losses = []
    for i in range(num_days):
        if i - last_week >= 5:
            pnl_change = pnl[i] - pnl[last_week]
            weekly_pnls.append(pnl_change)
            if pnl_change < 0:
                weekly_losses.append(pnl_change)
            last_week = i

    print('\nWeekly PnL Standard Deviation:', stdev(weekly_pnls))

    plt.figure()
    plt.hist(weekly_pnls, 50)
    plt.gca().set(title='Weekly PnL Distribution', xlabel='$',
                  ylabel='Frequency')

    # Sharpe ratio
    # Sharpe ratio is a very commonly used performance and risk metric that is
    # used in the industry to measure and compare the performance of algorithmic
    # trading strategies. Sharpe ratio is defined as the ratio of average PnL
    # over a period of time and the PnL standard deviation over the same period.
    # The benefit of the Sharpe ratio is that it captures the profitability of
    # a trading strategy while also accounting for the risk by using the
    # volatility of the returns. Another performance and risk measure similar
    # to the Sharpe Ratio is the Sortino Ratio, which only uses observations
    # where the trading strategy loses money and ignores the ones where the
    # trading strategy makes money. The idea is that, for a trading strategy,
    # Sharpe upside moves in PnLs are a good thing, so they should not be
    # considered when computing the standard deviation, i.e. only downside
    # moves or losses are actual risk observations. The following code computes
    # the Sharpe and Sortino Ratios for the trading strategy using a week as the
    # time horizon for the trading strategy.
    sharpe_ratio = mean(weekly_pnls) / stdev(weekly_pnls)
    sortino_ratio = mean(weekly_pnls) / stdev(weekly_losses)

    print('\nSharpe ratio:', sharpe_ratio)
    print('Sortino ratio:', sortino_ratio)

    # Maximum executions per period
    # This risk measure is an interval based risk check, which means it is a
    # counter that resets after a fixed amount of time and the risk check is
    # imposed within such a time slice. While there is no final limit, it's
    # important that the limit isn't exceeded within the time interval that is
    # meant to detect and avoid over trading. Maximum executions per period
    # measures the maximum number of trades allowed in a given timeframe. At the
    # end of the timeframe, the counter is reset and starts over. This is used
    # to detect and prevent a runaway trading strategy that buys and sells at a
    # very fast pace. The code below looks at the distribution of executions
    # for the trading strategy using a week as the timeframe.
    executions_this_week = 0
    executions_per_week = []
    last_week = 0
    for i in range(num_days):
        if results['Trades'].iloc[i] != 0:
            executions_this_week += 1

        if i - last_week >= 5:
            executions_per_week.append(executions_this_week)
            executions_this_week = 0
            last_week = i

    plt.figure()
    plt.hist(executions_per_week, 10)
    plt.gca().set(title='Weekly Number of Executions Distribution',
                  xlabel='Number of Executions', ylabel='Frequency')

    # Executions per month
    executions_this_month = 0
    executions_per_month = []
    last_month = 0
    for i in range(num_days):
        if results['Trades'].iloc[i] != 0:
            executions_this_month += 1

        if i - last_month >= 20:
            executions_per_month.append(executions_this_month)
            executions_this_month = 0
            last_month = i

    plt.figure()
    plt.hist(executions_per_month, 20)
    plt.gca().set(title='Monthly Number of Executions Distribution',
                  xlabel='Number of Executions', ylabel='Frequency')

    # Volume limits
    # This risk metric measures the traded volume, which can also have an
    # interval based variant that measures volume per period. This is another
    # risk measure that is meant to detect and prevent over trading. This risk
    # limit can be used to shut down trading strategies if it is exceeded. The
    # following code measures the trading volume which can be set as the volume
    # limit to detect over trading.
    traded_volume = 0
    for i in range(num_days):
        if results['Trades'].iloc[i] != 0:
            traded_volume += abs(results['Position'].iloc[i] -
                                 results['Position'].iloc[i - 1])
    print('\nTotal traded volume:', traded_volume)

    # Display plots and block
    plt.show()


if __name__ == '__main__':
    main()
