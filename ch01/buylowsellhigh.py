"""Buy Low Sell High trading strategy"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data


def main():
    start_date = '2014-01-01'
    end_date = '2018-01-01'
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)
    goog_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)
    print('Raw GOOG data from Yahoo Finance:')
    print(goog_data)
    goog_data_signal = pd.DataFrame(index=goog_data.index)
    # Use the adjusted closing price of the stock which is closing price
    # adjusted for corporate actions. Takes into account stock splits and
    # dividends.
    goog_data_signal['price'] = goog_data['Adj Close']
    # Signal based on price difference between consecutive days
    goog_data_signal['daily_difference'] = goog_data_signal['price'].diff()
    # Initialise signal column
    goog_data_signal['signal'] = 0.0
    # Trading signal: If the daily price difference is positive (price has
    # increased) then set signal=1.0 (sell) else set signal=0.0 (buy)
    goog_data_signal['signal'][:] = \
        np.where(goog_data_signal['daily_difference'][:] > 0, 1.0, 0.0)
    # Limit number of order by restricting to the number of positions on the
    # market to prevent constantly buying if the market keeps moving down, or
    # constantly selling when the market is moving up. Position is the
    # inventory of stocks or assets that we have on the market, e.g. if we
    # buy one Google share it means we have a position of one share on the
    # market. If we sell this share we will not have any positions on the
    # market.
    goog_data_signal['positions'] = goog_data_signal['signal'].diff()
    print('\nGoogle data signal:')
    print(goog_data_signal)

    # Plot the GOOG price
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Google price $')
    goog_data_signal['price'].plot(ax=ax1, color='r', lw=2.0)
    # Draw up arrow to indicate when we buy
    ax1.plot(goog_data_signal.loc[goog_data_signal.positions == 1.0].index,
             goog_data_signal.price[goog_data_signal.positions == 1.0],
             '^', markersize=5, color='m')
    # Draw a down arrow to indicate when we sell
    ax1.plot(goog_data_signal.loc[goog_data_signal.positions == -1.0].index,
             goog_data_signal.price[goog_data_signal.positions == -1.0],
             'v', markersize=5, color='k')
    plt.grid()

    # Backtesting - relies on the assumption that the past predicts the future
    # Test the strategy with an initial capital over a given period of time.
    initial_capital = 1000.0
    # Create DataFrame for the positions and the portfolio
    positions = pd.DataFrame(index=goog_data_signal.index).fillna(0.0)
    portfolio = pd.DataFrame(index=goog_data_signal.index).fillna(0.0)
    # Store GOOG positions
    positions['GOOG'] = goog_data_signal['signal']
    # Store the amount of GOOG positions in $ for the portfolio
    portfolio['positions'] = \
        positions.multiply(goog_data_signal['price'], axis=0)
    # Calculate the non-invested money (cash)
    portfolio['cash'] = \
        initial_capital - (positions.diff().multiply(goog_data_signal['price'],
                                                     axis=0)).cumsum()
    # Total investment is sum of positions in $ and cash
    portfolio['total'] = portfolio['positions'] + portfolio['cash']
    portfolio.plot()

    # Plot portfolio total value
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
    portfolio['total'].plot(ax=ax1, lw=2.0)
    ax1.plot(portfolio.loc[goog_data_signal.positions == 1.0].index,
             portfolio.total[goog_data_signal.positions == 1.0],
             '^', markersize=10, color='m')
    ax1.plot(portfolio.loc[goog_data_signal.positions == -1.0].index,
             portfolio.total[goog_data_signal.positions == -1.0],
             'v', markersize=10, color='k')
    plt.grid()

    # Display plots and block
    plt.show()


if __name__ == '__main__':
    main()
