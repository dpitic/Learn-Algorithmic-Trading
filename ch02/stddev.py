"""Standard Deviation
The standard deviation is a basic measure of price volatility that is used in
combination with a lot of other technical analysis indicators to improve them.
The standard deviation is a standard measure that is computed by measuring the
squared deviation of individual prices from the mean price, and then finding
the average of all those squared deviation values. This value is known as
variance, and the standard deviation is obtained by taking the square root of 
the variance. Larger standard deviations are a sign of more volatile markets or
larger expected price moves, so trading strategies need to factor that increased
volatility into risk estimates and other trading behaviour.
"""
import statistics as stats

import matplotlib.pyplot as plt
import pandas as pd

from algolib.data import get_google_data
from algolib.signals import standard_deviation


def main():
    # Get the Google data from Yahoo Finance from 2014-01-01 to 2018-01-01
    goog_data_raw = get_google_data()
    goog_data = goog_data_raw.tail(620)
    # Use close price for this analysis
    close = goog_data['Close']
    # Calculate standard deviation of close price over default time period = 20
    std_dev_list = standard_deviation(close)
    std_dev_df = pd.DataFrame(close)
    std_dev_df = std_dev_df.assign(
        std_dev=pd.Series(std_dev_list, index=close.index))
    print(std_dev_df)
    print('\nStatistical summary:')
    print(std_dev_df.describe())

    # Extract data to plot
    close_price = close
    std_dev = std_dev_df['std_dev']

    # Plot the data
    fig = plt.figure()
    ax1 = fig.add_subplot(211, ylabel='Google price in $')
    close_price.plot(ax=ax1, color='g', lw=2.0, legend=True)
    ax2 = fig.add_subplot(212, ylabel='Standard Deviation in $')
    std_dev.plot(ax=ax2, color='b', lw=2.0, legend=True)
    ax2.axhline(y=stats.mean(std_dev_list), color='k')

    ax1.grid()
    ax2.grid()
    plt.show()


if __name__ == "__main__":
    main()
