import statistics as stats

import numpy as np
import pandas as pd


def trading_support_resistance(data, bin_width=20):
    """Support and Resistance Trading Strategy

    A buy order is sent when a price stays in the resistance tolerance margin
    for 2 consecutive days, and a sell order when a price stays in the support
    tolerance margin for 2 consecutive days.

    :param DataFrame data: data signal.
    :param int bin_width: Number of days for rolling average.
    """
    data['sup_tolerance'] = pd.Series(np.zeros(len(data)))
    data['res_tolerance'] = pd.Series(np.zeros(len(data)))
    data['sup_count'] = pd.Series(np.zeros(len(data)))
    data['res_count'] = pd.Series(np.zeros(len(data)))
    data['sup'] = pd.Series(np.zeros(len(data)))
    data['res'] = pd.Series(np.zeros(len(data)))
    data['positions'] = pd.Series(np.zeros(len(data)))
    data['signal'] = pd.Series(np.zeros(len(data)))
    in_support = 0
    in_resistance = 0

    for x in range((bin_width - 1) + bin_width, len(data)):
        data_section = data[x - bin_width:x + 1]
        support_level = min(data_section['price'])
        resistance_level = max(data_section['price'])
        range_level = resistance_level - support_level
        data['res'][x] = resistance_level
        data['sup'][x] = support_level
        data['sup_tolerance'][x] = support_level + 0.2 * range_level
        data['res_tolerance'][x] = resistance_level - 0.2 * range_level

        if data['res_tolerance'][x] <= data['price'][x] <= data['res'][x]:
            in_resistance += 1
            data['res_count'][x] = in_resistance
        elif data['sup_tolerance'][x] >= data['price'][x] >= data['sup'][x]:
            in_support += 1
            data['sup_count'][x] = in_support
        else:
            in_support = 0
            in_resistance = 0

        if in_resistance > 2:
            data['signal'][x] = 1
        elif in_support > 2:
            data['signal'][x] = 0
        else:
            data['signal'][x] = data['signal'][x - 1]

    data['positions'] = data['signal'].diff()


def simple_moving_average(series, time_period=20):
    """Return Simple Moving Average (SMA) of the series.

    SMA is calculated by adding the price of an instrument over a number of time
    periods and then dividing the sum by the number of time periods. The SMA is
    basically the average price of the given time period, with equal weighting
    given to the price of each period.

    SMA = (sum(price, n)) / n

    Where: n = time period

    :param Series series: Price series.
    :param int time_period: Number of days over which to average, default=20
    :return: List of SMA prices.
    """
    history = []  # track history of prices
    sma_values = []  # track simple moving average values
    for price in series:
        history.append(price)
        # Remove oldest price because we only average over last time_period
        if len(history) > time_period:
            del history[0]
        sma_values.append(stats.mean(history))
    return sma_values


def exponential_moving_average(series, time_period=20):
    """Return Exponential Moving Average (EMA) of the series.

    EMA is similar to the Simple Moving Average (SMA) except instead of weighing
    all values equally, it places more weight on the most recent observations.

    EMA = (P - EMAp) * K + EMAp

    Where:
    P = Price for current period.
    EMAp = Exponential moving average for the previous period.
    K = smoothing constant, where K = 2 / (n + 1)
    n = number of time periods in a simple moving average roughly approximated
        by the EMA.

    :param Series series: Price series.
    :param int time_period: Number of days over which to average, default=20
    :return: List of EMA prices.
    """
    K = 2 / (time_period + 1)  # default smoothing constant
    ema_p = 0

    ema_values = []  # computed EMA values
    for price in series:
        if ema_p == 0:
            # First observation, EMA = current price
            ema_p = price
        else:
            ema_p = (price - ema_p) * K + ema_p
        ema_values.append(ema_p)
    return ema_values


def absolute_price_oscillator(series, time_period_fast=10, time_period_slow=20):
    """Return the Absolute Price Oscillator (APO) for the series.

    APO is the absolute difference between two moving averages of different
    lengths, a 'fast' and a 'slow' moving average.

    APO = EMAf - EMAs

    Where:
        EMAf is the fast exponential moving average.
        EMAs is the slow exponential moving average.
    :param Series series: Price series.
    :param int time_period_fast: Number of days over which to average the fast
        EMA, default=10.
    :param int time_period_slow: Number of days over which to average the slow
        EMA, default=20.
    :return: List of APO prices along with lists of EMAf and EMAs.
    """
    ema_fast_list = exponential_moving_average(series, time_period_fast)
    ema_fast = np.array(ema_fast_list)
    ema_slow_list = exponential_moving_average(series, time_period_slow)
    ema_slow = np.array(ema_slow_list)
    apo = ema_fast - ema_slow
    apo_list = apo.tolist()
    return apo_list, ema_fast_list, ema_slow_list


def moving_average_conv_div(series, time_period_fast=10, time_period_slow=40,
                            time_period_macd=20):
    """Return a Moving Average Convergence Divergence indicators.

    This function returns a DataFrame containing the following columns:
        Series: Original series passed into the function.
        EMA_fast: Fast exponential moving average.
        EMA_slow: Slow exponential moving average.
        MACD: Moving average convergence divergence.
        EMA_MACD: Exponential moving average of MACD.
        MACD_histogram: MACD histogram.
    The index of the DataFrame is the same as the index of the series passed
    into the function.

    :param Series series: Price series.
    :param int time_period_fast: Number of time periods of fast EMA, default=10
    :param int time_period_slow: Number of time periods of slow EMA, default=40
    :param int time_period_macd: Number of time periods of MACD, default=20.
    """
    K_fast = 2 / (time_period_fast + 1)
    ema_fast = 0
    K_slow = 2 / (time_period_slow + 1)
    ema_slow = 0
    K_macd = 2 / (time_period_macd + 1)
    ema_macd = 0

    ema_fast_list = []
    ema_slow_list = []
    macd_list = []
    macd_signal_list = []
    macd_histogram_list = []

    for price in series:
        if ema_fast == 0:
            ema_fast = price
            ema_slow = price
        else:
            ema_fast = (price - ema_fast) * K_fast + ema_fast
            ema_slow = (price - ema_slow) * K_slow + ema_slow

        ema_fast_list.append(ema_fast)
        ema_slow_list.append(ema_slow)

        macd = ema_fast - ema_slow
        if ema_macd == 0:
            ema_macd = macd
        else:
            ema_macd = (macd - ema_macd) * K_slow + ema_macd
            # ema_macd = (macd - ema_macd) * K_macd + ema_macd

        macd_list.append(macd)
        macd_signal_list.append(ema_macd)
        macd_histogram_list.append(macd - ema_macd)

    df = pd.DataFrame(series)
    df = df.assign(EMA_fast=pd.Series(ema_fast_list, index=series.index))
    df = df.assign(EMA_slow=pd.Series(ema_slow_list, index=series.index))
    df = df.assign(MACD=pd.Series(macd_list, index=series.index))
    df = df.assign(EMA_MACD=pd.Series(macd_signal_list, index=series.index))
    df = df.assign(
        MACD_histogram=pd.Series(macd_histogram_list, index=series.index))
    return df
