import math
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
    :return: DataFrame with various moving averages.
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


def bollinger_bands(series, time_period=20, std_dev_factor=2):
    """Return the Bollinger Bands.

    The Bollinger Bands provide the upper and lower envelope bands around the
    price of an instrument. The width of the bands is based on the standard
    deviation of the closing prices from a moving average price.

    Middle Band = n-period moving average
    Upper Band = Middle Band + (y * n-period standard deviation)
    Lower Band = Middle Band - (y * n-period standard deviation)

    Where:
        n = number of periods
        y = factor to apply to the standard deviation (typically y = 2)

    :param Series series: Price series.
    :param int time_period: Number of time periods for Simple Moving Average
        for middle band, default=20.
    :param int std_dev_factor: Standard deviation scaling factor for upper and
        lower bands.
    :return: DataFrame with price, middle, upper and lower Bollinger bands.
    """
    price_history_list = []  # price history for computing simple moving average
    price_sma_list = []  # moving average of prices
    upper_band_list = []  # upper band values
    lower_band_list = []  # lower band values

    for price in series:
        price_history_list.append(price)
        # Only maintain time_period number of price observations
        if len(price_history_list) > time_period:
            del price_history_list[0]

        sma = stats.mean(price_history_list)
        price_sma_list.append(sma)  # simple moving average or middle band
        variance = 0  # square of the standard deviation
        for hist_price in price_history_list:
            variance = variance + ((hist_price - sma) ** 2)

        stdev = math.sqrt(variance / len(price_history_list))

        upper_band_list.append(sma + std_dev_factor * stdev)
        lower_band_list.append(sma - std_dev_factor * stdev)

    df = pd.DataFrame(series)
    df = df.assign(MBBand=pd.Series(price_sma_list, index=series.index))
    df = df.assign(UBBand=pd.Series(upper_band_list, index=series.index))
    df = df.assign(LBBand=pd.Series(lower_band_list, index=series.index))
    return df


def relative_strength_index(series, time_period=20):
    """Return the Relative Strength Index (RSI).

    The current price is normalised as a percentage between 0 and 100. The RSI
    represents the current price relative to other recent prices within the
    selected lookback window length.

    RSI = 100 - (100 / (1 + RS))

    Where:
        RS = Ratio of smoothed average of n-period gains divided by the
             absolute value of the smoothed average of n-period losses.

    :param Series series: Price series.
    :param int time_periods: Lookback period to compute gains and losses.
    :return: List with price, average gains over lookback period, average
        loss over lookback period, and RSI values.
    """
    gain_history_list = []  # gains over look back period
    loss_history_list = []  # losses over look back period
    avg_gain_list = []  # average gains for visualisation purposes
    avg_loss_list = []  # average losses for visualisation purposes
    rsi_list = []  # computed RSI values
    # current_price - last price > 0 => gain
    # current_price - last price < 0 => loss
    last_price = 0

    for price in series:
        if last_price == 0:
            last_price = price

        gain_history_list.append(max(0, price - last_price))
        loss_history_list.append(max(0, last_price - price))
        last_price = price

        # Only keep gains and losses over look back time period for SMA calc.
        if len(gain_history_list) > time_period:
            del gain_history_list[0]
            del loss_history_list[0]

        # Gain and loss SMA over look back period
        avg_gain = stats.mean(gain_history_list)  # average gain over look back
        avg_loss = stats.mean(loss_history_list)  # average loss over look back

        avg_gain_list.append(avg_gain)
        avg_loss_list.append(avg_loss)

        rs = 0
        if avg_loss > 0:
            rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))
        rsi_list.append(rsi)

    df = pd.DataFrame(series)
    df = df.assign(RS_avg_gain=pd.Series(avg_gain_list, index=series.index))
    df = df.assign(RS_avg_loss=pd.Series(avg_loss_list, index=series.index))
    df = df.assign(RSI=pd.Series(rsi_list, index=series.index))
    return df


def standard_deviation(series, time_period=20):
    """Return the standard deviation over the specified time period.

    :param Series: series: Price series.
    :param int time_period: Look back period.
    :return: List of standard deviations calculated of over the time periods.
    """
    price_history_list = []  # history of prices for std. dev. calculation
    sma_values_list = []  # track moving average values for visualisation
    std_dev_list = []  # history of computed standard deviation values

    for price in series:
        price_history_list.append(price)
        # Only keep up to 'time_period' number of prices for std. dev. calc.
        if len(price_history_list) > time_period:
            del price_history_list[0]

        sma = stats.mean(price_history_list)
        sma_values_list.append(sma)
        variance = 0  # variance = square of standard deviation
        for hist_price in price_history_list:
            variance = variance + ((hist_price - sma) ** 2)

        std_dev = math.sqrt(variance / len(price_history_list))
        std_dev_list.append(std_dev)
    return std_dev_list


def momentum(series, time_period=20):
    """Return the Momentum indicator over the specified time period.

    The Momentum (MOM) indicator compares the current price with the previous
    price from a specified number of periods ago. This indicator is similar to
    the 'Rate of Change' indicator, but the MOM does not normalise the price,
    so different instruments can have different indicator values based on their
    point values.

    MOM = Price - Price of n periods ago

    :param Series series: Price series.
    :param int time_period: Amount of time to look back for reference price to
        compute momentum, default=20.
    :return: List of momentum values calculated over the time periods.
    """
    history_list = []  # historical observed prices to use in MOM calc.
    mom_list = []  # calculated MOM values

    for price in series:
        history_list.append(price)
        # Only use up to 'time_period' number of observations for calc.
        if len(history_list) > time_period:
            del history_list[0]

        mom = price - history_list[0]
        mom_list.append(mom)
    return mom_list
