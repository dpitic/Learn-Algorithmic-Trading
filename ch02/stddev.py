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

import matplotlib.pyplot as plt

import algolib.signals as signals
from algolib.data import get_google_data


def main():
    # Get the Google data from Yahoo Finance from 2014-01-01 to 2018-01-01
    goog_data_raw = get_google_data()
    # Use close price for this analysis
    close = goog_data_raw['Close']
    # Calculate standard deviation of SMA using default time period 20 days
    signals.avg_sma_std_dev(close, tail=620)
    plt.show()


if __name__ == "__main__":
    main()
