"""Moving Average Convergence Divergence
This is another class of indicators that builds on top of moving averages of
prices. Similar to APO, it establishes the difference between a fast exponential
moving average and a slow exponential moving average. However, to get the final
signal output from the MACD a smoothing exponential moving average is applied to
the MACD value itself.
"""
import matplotlib.pyplot as plt
import pandas as pd

from algolib.data import get_google_data
from algolib.signals import moving_average_conv_div


def main():
    # Get the Google data from Yahoo Finance from 2014-01-01 to 2018-01-01
    goog_data_raw = get_google_data()
    goog_data = goog_data_raw.tail(620)
    # Use close price for this analysis
    close = goog_data['Close']
    # Calculate the MACD using default time periods: fast=10; slow=40; macd=20
    macd_df = moving_average_conv_div(close)
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)
    print(macd_df)

    # Extract data to plot
    close_price = close
    ema_f = macd_df['EMA_fast']
    ema_s = macd_df['EMA_slow']
    macd = macd_df['MACD']
    ema_macd = macd_df['EMA_MACD']
    macd_histogram = macd_df['MACD_histogram']

    # Plot the data
    fig = plt.figure()
    ax1 = fig.add_subplot(311, ylabel='Google price in $')
    close_price.plot(ax=ax1, color='g', lw=2.0, legend=True)
    ema_f.plot(ax=ax1, color='b', lw=2.0, legend=True)
    ema_s.plot(ax=ax1, color='r', lw=2.0, legend=True)
    ax2 = fig.add_subplot(312, ylabel='MACD')
    macd.plot(ax=ax2, color='k', lw=2.0, legend=True)
    ema_macd.plot(ax=ax2, color='g', lw=2.0, legend=True)
    ax3 = fig.add_subplot(313, ylabel='MACD')
    macd_histogram.plot(ax=ax3, color='r', kind='bar', legend=True,
                        use_index=False)
    ax1.grid()
    ax2.grid()
    plt.show()


if __name__ == '__main__':
    main()
