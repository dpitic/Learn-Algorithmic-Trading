"""Absolute Price Oscillator
This indicator builds on top of moving averages of prices to capture specific
short-term deviations in price.  The APO is the difference between a fast
exponential moving average and a slow exponential moving average.  It is trying
to measure how for the more reactive fast EMA is deviating from the more stable
slow EMA.  A large difference usually means one of two things:
1. Instrument prices are starting to trend or break out, or
2. Instrument prices are far away from their equilibrium prices, i.e.
   overbought or oversold.
This module implements an APO with the faster EMA using a period of 10 days and
the slower EMA using a period of 40 days.
"""
import matplotlib.pyplot as plt
import pandas as pd

from algolib.data import get_google_data
from algolib.signals import absolute_price_oscillator


def main():
    # Get the Google data from Yahoo Finance from 2014-01-01 to 2018-01-01
    goog_data_raw = get_google_data()
    goog_data = goog_data_raw.tail(620)
    # Use close price for this analysis
    close = goog_data['Close']
    close_apo, close_fast_ema, close_slow_ema = \
        absolute_price_oscillator(close, time_period_fast=10,
                                  time_period_slow=40)
    goog_data = goog_data.assign(
        ClosePrice=pd.Series(close, index=goog_data.index))
    goog_data = goog_data.assign(
        FastExponential10DayMovingAverage=pd.Series(close_fast_ema,
                                                    index=goog_data.index))
    goog_data = goog_data.assign(
        SlowExponential40DayMovingAverage=pd.Series(close_slow_ema,
                                                    index=goog_data.index))
    goog_data = goog_data.assign(
        AbsolutePriceOscillator=pd.Series(close_apo, index=goog_data.index))

    close_price = goog_data['ClosePrice']
    ema_fast = goog_data['FastExponential10DayMovingAverage']
    ema_slow = goog_data['SlowExponential40DayMovingAverage']
    apo = goog_data['AbsolutePriceOscillator']

    # Plot the close price, APO and fast and slow EMAs
    fig = plt.figure()
    ax1 = fig.add_subplot(211, ylabel='Google price in $')
    close_price.plot(ax=ax1, color='g', lw=2.0, legend=True)
    ema_fast.plot(ax=ax1, color='b', lw=2.0, legend=True)
    ema_slow.plot(ax=ax1, color='r', lw=2.0, legend=True)
    ax2 = fig.add_subplot(212, ylabel='APO')
    apo.plot(ax=ax2, color='k', lw=2.0, legend=True)
    ax1.grid()
    ax2.grid()
    plt.show()


if __name__ == '__main__':
    main()
