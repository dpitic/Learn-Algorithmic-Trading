"""Bollinger Bands
This indicator builds on top of moving averages, but incorporates a recent price
volatility that makes the indicator more adaptive to different market 
conditions. The indicator computes a moving average of the prices, which can
be a simple moving average or the exponential moving average or any other
moving average. It also calculates the standard deviation of the prices in the
lookback period by treating the moving average as the mean price. It then
creates an upper band that is a moving average plus some multiple of standard
deviations, and a lower band that is a moving average minus some multiple of
the standard price deviation. This band represents the expected volatility of
the prices by treating the moving average of the price as the reference price.
When prices move outside of these bands, it can be interpreted as a 
breakout/trend signal or an overbought/sold mean reversion signal.
"""
import matplotlib.pyplot as plt
import pandas as pd

from algolib.data import get_google_data
from algolib.signals import bollinger_bands


def main():
    # Get the Google data from Yahoo Finance from 2014-01-01 to 2018-01-01
    goog_data_raw = get_google_data()
    goog_data = goog_data_raw.tail(620)
    # Use close price for this analysis
    close = goog_data['Close']
    # Calculate Bollinger Bands: default time period = 20; standard deviation
    # factor = 2
    bbands_df = bollinger_bands(close)
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)
    print(bbands_df)

    # Extract data to plot
    close_price = close
    mband = bbands_df['MBBand']
    uband = bbands_df['UBBand']
    lband = bbands_df['LBBand']

    # Plot the data
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Google price in $')
    close_price.plot(ax=ax1, color='c', lw=2.0, legend=True)
    mband.plot(ax=ax1, color='b', lw=2.0, legend=True)
    uband.plot(ax=ax1, color='g', lw=2.0, legend=True)
    lband.plot(ax=ax1, color='r', lw=2.0, legend=True)

    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
