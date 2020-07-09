"""Momentum
This is an important measure of speed and magnitude of price moves and is often
a key indicator of trend/breakout-based trading algorithms. Momentum is the
difference between the current price and price of some fixed time periods in
the past. Consecutive periods of positive momentum values indicate an uptrend;
consecutive periods of negative momentum values indicate a downtrend. Often the
simple/exponential moving averages of the momentum indicator are used to
detect sustained trends.
"""
import matplotlib.pyplot as plt
import pandas as pd

from algolib.data import get_google_data
from algolib.signals import momentum


def main():
    # Get the Google data from Yahoo Finance from 2014-01-01 to 2018-01-01
    goog_data_raw = get_google_data()
    goog_data = goog_data_raw.tail(620)
    # Use close price for this analysis
    close = goog_data['Close']
    # Calculate momentum of close price using default time period = 20
    mom_list = momentum(close)
    mom_df = pd.DataFrame(close)
    mom_df = mom_df.assign(mom=pd.Series(mom_list, index=close.index))
    print(mom_df)

    # Extract data to plot
    close_price = close
    mom = mom_df['mom']

    # Plot the data
    fig = plt.figure()
    ax1 = fig.add_subplot(211, ylabel='Google price in $')
    close_price.plot(ax=ax1, color='g', lw=2.0, legend=True)
    ax2 = fig.add_subplot(212, ylabel='Momentum in $')
    mom.plot(ax=ax2, color='b', lw=2.0, legend=True)

    ax1.grid()
    ax2.grid()
    plt.show()


if __name__ == "__main__":
    main()
