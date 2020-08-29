import math
import statistics as stats

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


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
    :return: DataFrame of original price series, apo=APO, ema_fast=EMA fast
        and ema_slow=EMA slow.
    """
    ema_fast_list = exponential_moving_average(series, time_period_fast)
    ema_fast = np.array(ema_fast_list)
    ema_slow_list = exponential_moving_average(series, time_period_slow)
    ema_slow = np.array(ema_slow_list)
    apo = ema_fast - ema_slow

    df = pd.DataFrame(series)
    df = df.assign(apo=pd.Series(apo, index=series.index))
    df = df.assign(ema_fast=pd.Series(ema_fast, index=series.index))
    df = df.assign(ema_slow=pd.Series(ema_slow, index=series.index))
    return df


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
            # ema_macd = (macd - ema_macd) * K_slow + ema_macd
            ema_macd = (macd - ema_macd) * K_macd + ema_macd

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


def dual_moving_average(financial_data, short_window, long_window):
    """Return Dual Moving Average trading signals using close price.

    This trading signal uses an additional moving average to limit the number of
    switches. It uses a short-ter moving average and a long-term moving average.
    With this implementation, the momentum shifts in the direction of the
    short-term moving average. When the short-term moving average crosses the
    long-term moving average and its value exceeds that of the long-term moving
    average, the momentum will be upward and this can lead to the adoption of a
    long position. If the momentum is in the opposite direction, this can lead
    to take a short position instead.

    This function returns a DataFrame with four columns.
        signal:     1 if the signal is going long and 0 if it is going short.
        short_mavg: Short-term moving average.
        long_mavg:  Long-term moving average.
        orders:     the side of the orders where buy=1 & sell=-1.

    :param DataFrame or Series financial_data: Close price.
    :param int short_window: Short-term moving average window.
    :param int long_window: Long-term moving average window.
    :return DataFrame: Dual moving average trading signals.
    """
    signals = pd.DataFrame(index=financial_data.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = financial_data['Close'].rolling(
        window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = financial_data['Close'].rolling(
        window=long_window, min_periods=1, center=False).mean()
    signals['signal'][short_window:] = \
        np.where(signals['short_mavg'][short_window:] >
                 signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['orders'] = signals['signal'].diff()
    return signals


def naive_momentum_trading(financial_data, num_conseq_days):
    """Naive momentum based trading strategy using adjusted close price.

    This strategy is based on the number of times a price increases or 
    decreases using the historical price momentum of the adjusted
    close price. It counts the number of times a price is improved:
      * If the number is equal to a given threshold, a buy signal is recorded,
        assuming the price will keep rising.
      * A sell signal is recorder assuming the price will keep dropping.

    :param DataFrame or Series financial_data: Adjusted close price.
    :param int num_conseq_days: Threshold for number of consecutive days.
    :return: Trading signals DataFrame where buy=1, sell=-1.
    """
    signals = pd.DataFrame(index=financial_data.index)
    signals['orders'] = 0
    cons_days = 0
    prior_price = 0
    init = True
    for k in range(len(financial_data['Adj Close'])):
        price = financial_data['Adj Close'][k]
        if init:
            prior_price = price
            init = False
        elif price > prior_price:
            if cons_days < 0:
                cons_days = 0
            cons_days += 1
        elif price < prior_price:
            if cons_days > 0:
                cons_days = 0
            cons_days -= 1

        if cons_days == num_conseq_days:
            signals['orders'][k] = 1
        elif cons_days == -num_conseq_days:
            signals['orders'][k] = -1

    return signals


def turtle_trading(financial_data, window_size):
    """Turtle Trading Strategy.

    This trading strategy creates a long (buy) signal when the price reaches the
    highest price for the last number of days specified by the window_size. A
    short (sell) signal is created when the price reaches its lowest point. It
    gets out of a position by having the price crossing the moving average of
    the last window_size number of days.

    :param DataFrame or Series financial_data: Adjusted close price.
    :param int window_size: Number of days threshold.
    :return: Trading signal DataFrame with the following columns:
        orders:      long position (buy) is 1; short position (sell) is -1;
                     do nothing is 0.
        high:        window_size days high.
        low:         window_size days low.
        avg:         window_size days rolling average.
        long_entry:  stock price > highest value for window_size days
        short_entry: stock price < lowest value for window_size days
        long_exit:   stock price crosses the mean of past window_size days
        short_exit:  stock price crosses the mean of past window_size days
    """
    signals = pd.DataFrame(index=financial_data.index)
    signals['orders'] = 0
    # window_size days high
    signals['high'] = \
        financial_data['Adj Close'].shift(1).rolling(window=window_size).max()
    # window_size days low
    signals['low'] = \
        financial_data['Adj Close'].shift(1).rolling(window=window_size).min()
    # window_size days mean
    signals['avg'] = \
        financial_data['Adj Close'].shift(1).rolling(window=window_size).mean()

    # Rules to place an order (entry rule):
    #     stock price > highest value for window_size days
    #     stock price < lowest value for window_size days
    signals['long_entry'] = financial_data['Adj Close'] > signals.high
    signals['short_entry'] = financial_data['Adj Close'] < signals.low

    # Rule to get out of a position (exit rule):
    #     stock price crosses the mean of past window_size days
    signals['long_exit'] = financial_data['Adj Close'] < signals.avg
    signals['short_exit'] = financial_data['Adj Close'] > signals.avg

    # Orders are represented by long position (buy=1), short position (sell=-1)
    # and 0 for not changing anything
    init = True
    position = 0
    for k in range(len(signals)):
        if signals['long_entry'][k] and position == 0:
            signals.orders.values[k] = 1  # long position (buy)
            position = 1
        elif signals['short_entry'][k] and position == 0:
            signals.orders.values[k] = -1  # short position (sell)
            position = -1
        elif signals['short_exit'][k] and position > 0:
            signals.orders.values[k] = -1  # short position (sell)
            position = 0
        elif signals['long_exit'][k] and position < 0:
            signals.orders.values[k] = 1  # long position (buy)
            position = 0
        else:
            signals.orders.values[k] = 0

    return signals


def find_cointegrated_pairs(data):
    """Return p-value matrix and list of cointegrated pairs.

    This function establishes cointegration between pairs from a DataFrame of
    financial instruments and calculates the cointegration values of these
    symbols.
    :param DataFrame data: Trading symbols.
    :return: p-value matrix and list of pairs.
    """
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            result = coint(data[keys[i]], data[keys[j]])
            pvalue_matrix[i, j] = result[1]
            if result[1] < 0.02:
                pairs.append((keys[i], keys[j]))
    return pvalue_matrix, pairs


def zscore(series):
    """Return how far a piece of data is from the population mean.

    The z-score helps determine the direction of trading. If the return value
    is positive, the symbol price is higher than the average price value.
    Therefore, its price is expected to go down or the paired symbol value will
    go up. In this we want to short this symbol and long the other one.
    """
    return (series - series.mean()) / np.std(series)
