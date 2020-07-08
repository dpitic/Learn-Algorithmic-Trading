"""Relative Strength Index (RSI)
This indicator is based on price changes over periods to capture the strength or
magnitude of price moves. It comprises a lookback period, which it uses to
compute the magnitude of the average gains/price increases over that period, as
well as the magnitude of the averages of losses/price decreases over that
period. Then it computes the RSI value that normalises the signal value to stay
between 0 and 100, and attempts to capture if there have been many more gains
relative to the losses, or if there have been many more losses relative to the
gains. RSI values over 50% indicate an uptrend, while values below 50% indicate
a downtrend.
"""
import matplotlib.pyplot as plt
import pandas as pd

from algolib.data import get_google_data
from algolib.signals import relative_strength_index


def main():
    # Get the Google data from Yahoo Finance from 2014-01-01 to 2018-01-01
    goog_data_raw = get_google_data()
    goog_data = goog_data_raw.tail(620)
    # Use close price for this analysis
    close = goog_data['Close']
    # Calculate Relative Strength Index using default time period = 20
    rsi_df = relative_strength_index(close)
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)
    print(rsi_df)

    # Extract data to plot
    close_price = close
    rs_gain = rsi_df['RS_avg_gain']
    rs_loss = rsi_df['RS_avg_loss']
    rsi = rsi_df['RSI']

    # Plot the data
    fig = plt.figure()
    ax1 = fig.add_subplot(311, ylabel='Google price in $')
    close_price.plot(ax=ax1, color='k', lw=2.0, legend=True)
    ax2 = fig.add_subplot(312, ylabel='RS')
    rs_gain.plot(ax=ax2, color='g', lw=2.0, legend=True)
    rs_loss.plot(ax=ax2, color='r', lw=2.0, legend=True)
    ax3 = fig.add_subplot(313, ylabel='RSI')
    rsi.plot(ax=ax3, color='b', lw=2.0, legend=True)

    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.show()


if __name__ == '__main__':
    main()
