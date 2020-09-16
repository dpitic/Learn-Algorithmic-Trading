import math
import statistics as stats
from itertools import cycle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
    """Return the standard deviation of the SMA over the specified time period.

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


def avg_sma_std_dev(prices, tail=None, sma_time_periods=20):
    """Return average standard deviation using SMA.

    This function calculates the standard deviation of tail number of prices
    using the simple moving average over the specified number of time periods.
    It also plots the prices and the standard deviation of the prices based on
    the simple moving average calculation.

    :param Series prices: Price series.
    :param tail: Tail number of prices to use, default=None.
    :param int sma_time_periods: Number of time periods to use in SMA
        calculation used to calculate the standard deviation of prices,
        default=20.
    :return: Average standard deviation using SMA.
    """
    if tail is not None:
        tail_close = prices.tail(tail)
    else:
        tail_close = prices
    std_dev_list = standard_deviation(tail_close, sma_time_periods)
    tail_close_df = pd.DataFrame(tail_close)
    tail_close_df = tail_close_df.assign(
        std_dev=pd.Series(std_dev_list, index=tail_close.index))
    print(f'Last {tail} close prices and standard deviation of '
          f'{sma_time_periods} days SMA:')
    print(tail_close_df)
    print('\nStatistical summary:')
    print(tail_close_df.describe())
    # Average standard deviation of prices SMA over look back period
    avg_std_dev = stats.mean(std_dev_list)
    # Extract data to plot
    close_price = tail_close
    std_dev = tail_close_df['std_dev']
    # Plot last tail number prices and standard deviation
    fig = plt.figure()
    ax1 = fig.add_subplot(211, ylabel='Google price in $')
    close_price.plot(ax=ax1, color='g', lw=2.0, legend=True, grid=True)
    ax2 = fig.add_subplot(212, ylabel='Standard Deviation in $')
    std_dev.plot(ax=ax2, color='b', lw=2.0, legend=True, grid=True)
    # Plot average standard deviation of SMA
    ax2.axhline(y=avg_std_dev, color='k')
    return avg_std_dev


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


def basic_mean_reversion(prices, ema_time_period_fast=10,
                         ema_time_period_slow=40, apo_value_for_buy_entry=-10,
                         apo_value_for_sell_entry=10,
                         min_price_move_from_last_trade=10,
                         num_shares_per_trade=10, min_profit_to_close=10):
    """Return mean reversion trading strategy results based on APO.

    This function implements a mean reversion trading strategy that relies on
    the Absolute Price Oscillator (APO) trading signal. By default it uses
    10 days for the fast EMA and 40 days for the slow EMA. It will perform
    buy trades when the APO signal value drops below -10 and perform sell trades
    when the APO signal value goes above +10. It will check that new trades are
    made at prices that are different from the last trade price to prevent over
    trading. Positions are closed when the APO signal value changes sign:
        * Close short positions when the APO goes negative, and
        * Close long positions when the APO goes positive.
    Positions are also closed if current open positions are profitable above a
    certain amount, regardless of the APO values. This is used to
    algorithmically lock profits and initiate more positions instead of relying
    on the trading signal value.

    :param Series prices: Price series.
    :param int ema_time_period_fast: Number of time periods for fast EMA,
        default=10.
    :param int ema_time_period_slow: Number of time periods for slow EMA,
        default=40.
    :param int apo_value_for_buy_entry: APO trading signal value below which to
        enter buy orders/long positions, default = -10.
    :param int apo_value_for_sell_entry: APO trading signal value above which to
        enter sell orders/short positions, default=10.
    :param int min_price_move_from_last_trade: Minimum price change since last
        trade before considering trading again. This prevents over trading at
        around the same prices, default=10.
    :param int num_shares_per_trade: Number of shares to buy/sell on every
        trade, default=10.
    :param int min_profit_to_close: Minimum open/unrealised profit at which to
         close and lock profits, default=10.
    :return: DataFrame containing the following columns:
        ClosePrice = price series provided as a parameter to the function.
        FastEMA = Fast exponential moving average.
        SlowEMA = Slow exponential moving average.
        APO = Absolute price oscillator.
        Trades = Buy/sell orders: buy=+1; sell=-1; no action=0.
        Positions = Long=+ve; short=-ve, flat/no position=0.
        PnL = Profit and loss.
    """
    # Variables for trading strategy trade, position and p&l management

    # Track buy/sell orders: buy=+1, sell=-1, no action=0
    orders = []
    # Track positions: long=+ve, short=-ve, flat/no position=0
    positions = []
    # Track total p&l
    pnls = []
    # Price at which last buy trade was made; used to prevent over trading
    last_buy_price = 0
    # Price at which last sell trade was made; used to prevent over trading
    last_sell_price = 0
    # Current position of the trading strategy
    position = 0
    # Sum of buy_trade_price and buy_trade_qty for every buy trade made since
    # last time being flat
    buy_sum_price_qty = 0
    # Summation of buy_trade_qty for every buy trade made since last time being
    # flat
    buy_sum_qty = 0
    # Sum of products of sell_trade_price and sell_trade_qty for every sell
    # trade made since last time being flat
    sell_sum_price_qty = 0
    # Sum of sell_trade_qty for every sell Trade made since last time being
    # flat
    sell_sum_qty = 0
    # Open/unrealised PnL marked to market
    open_pnl = 0
    # Closed/realised PnL so far
    closed_pnl = 0

    # Trading strategy

    # Calculate fast and slow EMA and APO on close price
    apo_df = absolute_price_oscillator(prices,
                                       time_period_fast=ema_time_period_fast,
                                       time_period_slow=ema_time_period_slow)
    ema_fast_values = apo_df.loc[:, 'ema_fast'].tolist()
    ema_slow_values = apo_df.loc[:, 'ema_slow'].tolist()
    apo_values = apo_df.loc[:, 'apo'].tolist()

    # Trading strategy main loop
    for close_price, apo in zip(prices, apo_values):
        # Check trading signal against trading parameters/thresholds and
        # positions to trade

        # Perform a sell trade at close_price on the following conditions:
        # 1. APO trading signal value is above sell entry threshold and the
        #    difference between last trade price and current price is different
        #    enough, or
        # 2. We are long (+ve position) and either APO trading signal value is
        #    at or above 0 or current position is profitable enough to lock
        #    profit.
        if ((apo > apo_value_for_sell_entry and abs(
                close_price - last_sell_price) > min_price_move_from_last_trade)
                or
                (position > 0 and (apo >= 0 or
                                   open_pnl > min_profit_to_close))):
            orders.append(-1)  # mark the sell trade
            last_sell_price = close_price
            position -= num_shares_per_trade  # reduce position by size of trade
            sell_sum_price_qty += close_price * num_shares_per_trade
            sell_sum_qty += num_shares_per_trade
            print('Sell ', num_shares_per_trade, ' @ ', close_price,
                  'Position: ', position)

        # Perform a buy trade at close_price on the following conditions:
        # 1. APO trading signal value is below buy entry threshold and the
        #    difference between last trade price and current price is different
        #    enough, or
        # 2. We are short (-ve position) and either APO trading signal value is
        #    at or below 0 or current position is profitable enough to lock
        #    profit.
        elif ((apo < apo_value_for_buy_entry and abs(
                close_price - last_buy_price) > min_price_move_from_last_trade)
              or
              (position < 0 and (apo <= 0 or open_pnl > min_profit_to_close))):
            orders.append(+1)  # mark the buy trade
            last_buy_price = close_price
            position += num_shares_per_trade  # increase position by trade size
            buy_sum_price_qty += close_price * num_shares_per_trade
            buy_sum_qty += num_shares_per_trade
            print('Buy ', num_shares_per_trade, ' @ ', close_price,
                  'Position: ', position)
        else:
            # No trade since none of the conditions were met to buy or sell
            orders.append(0)

        positions.append(position)

        # Update open/unrealised and closed/realised positions
        open_pnl = 0
        if position > 0:
            if sell_sum_qty > 0:
                # Long position and some sell trades have been made against it,
                # close that amount based on how much was sold against this
                # long position.
                open_pnl = abs(sell_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # would be if we closed at current price.
            open_pnl += abs(sell_sum_qty - position) * (
                    close_price - buy_sum_price_qty / buy_sum_qty)
        elif position < 0:
            if buy_sum_qty > 0:
                # Short position and some buy trades have been made against it,
                # close that amount based on how much was bought against this
                # short position.
                open_pnl = abs(buy_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # wold be if we closed at current price
            open_pnl += abs(buy_sum_qty - position) * (
                    sell_sum_price_qty / sell_sum_qty - close_price)
        else:
            # Flat, so update closed pnl and reset tracking variables for
            # positions and pnls
            closed_pnl += sell_sum_price_qty - buy_sum_price_qty
            buy_sum_price_qty = 0
            buy_sum_qty = 0
            sell_sum_price_qty = 0
            sell_sum_qty = 0
            last_buy_price = 0
            last_sell_price = 0

        print('OpenPnL: ', open_pnl, ' ClosedPnL: ', closed_pnl,
              ' TotalPnL', (open_pnl + closed_pnl))
        pnls.append(closed_pnl + open_pnl)

    # Prepare DataFrame from the trading strategy results
    df = prices.to_frame(name='ClosePrice')
    df = df.assign(FastEMA=pd.Series(ema_fast_values, index=df.index))
    df = df.assign(SlowEMA=pd.Series(ema_slow_values, index=df.index))
    df = df.assign(APO=pd.Series(apo_values, index=df.index))
    df = df.assign(Trades=pd.Series(orders, index=df.index))
    df = df.assign(Position=pd.Series(positions, index=df.index))
    df = df.assign(PnL=pd.Series(pnls, index=df.index))
    return df


def volatility_mean_reversion(prices, sma_time_periods=20, avg_std_dev=None,
                              ema_time_period_fast=10, ema_time_period_slow=40,
                              apo_value_for_buy_entry=-10,
                              apo_value_for_sell_entry=10,
                              min_price_move_from_last_trade=10,
                              num_shares_per_trade=10, min_profit_to_close=10):
    """Return mean reversion trading strategy based on volatility adjusted APO.

    This function uses the standard deviation as a volatility measure to
    adjust the  number of days used in the fast and slow EAM to produce a
    volatility adjusted APO entry signal. By default it uses 10 days for the
    fast EMA and 40 days for the slow EMA. It will perform buy trades when the
    APO signal value drops below -10 and perform sell trades when the APO
    signal value goes above +10. It will check that new trades are made at
    prices that are different from the last trade price to prevent over trading.
    Positions are closed when the APO signal value changes sign:
        * Close short positions when the APO goes negative, and
        * Close long positions when the APO goes positive.
    Positions are also closed if current open positions are profitable above a
    certain amount, regardless of the APO values. This is used to
    algorithmically lock profits and initiate more positions instead of relying
    on the trading signal value.

    :param Series prices: Price series.
    :param int sma_time_periods: Simple moving average look back period,
        default=20.
    :param avg_std_dev: Average standard deviation of prices SMA over
        look back period of sma_time_periods number of days. If this is not
        specified, it is calculated by the function, default=None.
    :param int ema_time_period_fast: Number of time periods for fast EMA,
        default=10.
    :param int ema_time_period_slow: Number of time periods for slow EMA,
        default=40.
    :param int apo_value_for_buy_entry: APO trading signal value below which to
        enter buy orders/long positions, default = -10.
    :param int apo_value_for_sell_entry: APO trading signal value above which to
        enter sell orders/short positions, default=10.
    :param int min_price_move_from_last_trade: Minimum price change since last
        trade before considering trading again. This prevents over trading at
        around the same prices, default=10.
    :param int num_shares_per_trade: Number of shares to buy/sell on every
        trade, default=10.
    :param int min_profit_to_close: Minimum open/unrealised profit at which to
         close and lock profits, default=10.
    :return: DataFrame containing the following columns:
        ClosePrice = price series provided as a parameter to the function.
        FastEMA = Fast exponential moving average.
        SlowEMA = Slow exponential moving average.
        APO = Absolute price oscillator.
        Trades = Buy/sell orders: buy=+1; sell=-1; no action=0.
        Positions = Long=+ve; short=-ve, flat/no position=0.
        PnL = Profit and loss.
    """
    # Variables for EMA calculation
    k_fast = 2 / (ema_time_period_fast + 1)  # fast EMA smoothing factor
    ema_fast = 0
    ema_fast_list = []  # calculated fast EMA values

    k_slow = 2 / (ema_time_period_slow + 1)  # slow EMA smoothing factor
    ema_slow = 0
    ema_slow_list = []  # calculated slow EMA values

    apo_list = []  # calculated absolute price oscillated signals

    # Variables for trading strategy trade, position and p&l management

    # Track buy/sell orders: buy=+1, sell=-1, no action=0
    orders = []
    # Track positions: long=+ve, short=-ve, flat/no position=0
    positions = []
    # Track total p&l
    pnls = []
    # Price at which last buy trade was made; used to prevent over trading
    last_buy_price = 0
    # Price at which last sell trade was made; used to prevent over trading
    last_sell_price = 0
    # Current position of the trading strategy
    position = 0
    # Sum of buy_trade_price and buy_trade_qty for every buy trade made since
    # last time being flat
    buy_sum_price_qty = 0
    # Summation of buy_trade_qty for every buy trade made since last time being
    # flat
    buy_sum_qty = 0
    # Sum of products of sell_trade_price and sell_trade_qty for every sell
    # trade made since last time being flat
    sell_sum_price_qty = 0
    # Sum of sell_trade_qty for every sell Trade made since last time being
    # flat
    sell_sum_qty = 0
    # Open/unrealised PnL marked to market
    open_pnl = 0
    # Closed/realised PnL so far
    closed_pnl = 0

    # Price history over sma_time_periods number of time periods for SMA and
    # standard deviation calculation used as a volatility measure
    price_history = []

    # Calculate average standard deviation of prices SMA if required
    if avg_std_dev is None:
        std_dev_list = standard_deviation(prices, time_period=sma_time_periods)
        avg_std_dev = stats.mean(std_dev_list)

    # Trading strategy main loop
    for close_price in prices:
        price_history.append(close_price)
        # Only track at most sma_time_periods number of prices
        if len(price_history) > sma_time_periods:
            del price_history[0]

        # Calculated SMA over sma_time_periods number of days
        sma = stats.mean(price_history)
        # Calculate variance over sma_time_periods number of days
        variance = 0  # variance is square of standard deviation
        for hist_price in price_history:
            variance += (hist_price - sma) ** 2

        stdev = math.sqrt(variance / len(price_history))
        stdev_factor = stdev / avg_std_dev
        if stdev_factor == 0:
            stdev_factor = 1

        # Calculate the fast and slow EMAs with smoothing factors adjusted for
        # volatility
        if ema_fast == 0:  # first observation
            ema_fast = close_price
            ema_slow = close_price
        else:
            ema_fast = (close_price - ema_fast) \
                       * k_fast * stdev_factor + ema_fast
            ema_slow = (close_price - ema_slow) \
                       * k_slow * stdev_factor + ema_slow

        ema_fast_list.append(ema_fast)
        ema_slow_list.append(ema_slow)

        # Calculate APO trading signal based on volatility adjusted EMAs
        apo = ema_fast - ema_slow
        apo_list.append(apo)

        # Check trading signal against trading parameters/thresholds and
        # positions to trade. This code uses dynamic thresholds based on
        # volatility for APO buy and sell entry thresholds. This makes the
        # strategy less aggressive in entering positions during periods of
        # higher volatility by increasing the threshold for entry by a factor
        # of volatility. Additionally, volatility is incorporated in the
        # expected profit threshold to lock in profit in a position by having
        # a dynamic threshold based on volatility.

        # Perform a sell trade at close_price on the following conditions:
        # 1. APO trading signal value is above sell entry threshold and the
        #    difference between last trade price and current price is different
        #    enough, or
        # 2. We are long (+ve position) and either APO trading signal value is
        #    at or above 0 or current position is profitable enough to lock
        #    profit.
        if ((apo > apo_value_for_sell_entry * stdev_factor and abs(
                close_price - last_sell_price) >
             min_price_move_from_last_trade * stdev_factor)
                or
                (position > 0 and (apo >= 0 or open_pnl >
                                   min_profit_to_close / stdev_factor))):
            orders.append(-1)  # mark the sell trade
            last_sell_price = close_price
            position -= num_shares_per_trade  # reduce position by size of trade
            sell_sum_price_qty += close_price * num_shares_per_trade
            sell_sum_qty += num_shares_per_trade
            print('Sell ', num_shares_per_trade, ' @ ', close_price,
                  'Position: ', position)

        # Perform a buy trade at close_price on the following conditions:
        # 1. APO trading signal value is below buy entry threshold and the
        #    difference between last trade price and current price is different
        #    enough, or
        # 2. We are short (-ve position) and either APO trading signal value is
        #    at or below 0 or current position is profitable enough to lock
        #    profit.
        elif ((apo < apo_value_for_buy_entry * stdev_factor and abs(
                close_price - last_buy_price) >
               min_price_move_from_last_trade * stdev_factor)
              or
              (position < 0 and (apo <= 0 or open_pnl >
                                 min_profit_to_close / stdev_factor))):
            orders.append(+1)  # mark the buy trade
            last_buy_price = close_price
            position += num_shares_per_trade  # increase position by trade size
            buy_sum_price_qty += close_price * num_shares_per_trade
            buy_sum_qty += num_shares_per_trade
            print('Buy ', num_shares_per_trade, ' @ ', close_price,
                  'Position: ', position)
        else:
            # No trade since none of the conditions were met to buy or sell
            orders.append(0)

        positions.append(position)

        # Update open/unrealised and closed/realised positions
        open_pnl = 0
        if position > 0:
            if sell_sum_qty > 0:
                # Long position and some sell trades have been made against it,
                # close that amount based on how much was sold against this
                # long position.
                open_pnl = abs(sell_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # would be if we closed at current price.
            open_pnl += abs(sell_sum_qty - position) * (
                    close_price - buy_sum_price_qty / buy_sum_qty)
        elif position < 0:
            if buy_sum_qty > 0:
                # Short position and some buy trades have been made against it,
                # close that amount based on how much was bought against this
                # short position.
                open_pnl = abs(buy_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # wold be if we closed at current price
            open_pnl += abs(buy_sum_qty - position) * (
                    sell_sum_price_qty / sell_sum_qty - close_price)
        else:
            # Flat, so update closed pnl and reset tracking variables for
            # positions and pnls
            closed_pnl += sell_sum_price_qty - buy_sum_price_qty
            buy_sum_price_qty = 0
            buy_sum_qty = 0
            sell_sum_price_qty = 0
            sell_sum_qty = 0
            last_buy_price = 0
            last_sell_price = 0

        print('OpenPnL:', open_pnl, ' ClosedPnL:', closed_pnl,
              ' TotalPnL:', (open_pnl + closed_pnl))
        pnls.append(closed_pnl + open_pnl)

    # Prepare DataFrame from the trading strategy results
    df = prices.to_frame(name='ClosePrice')
    df = df.assign(FastEMA=pd.Series(ema_fast_list, index=df.index))
    df = df.assign(SlowEMA=pd.Series(ema_slow_list, index=df.index))
    df = df.assign(APO=pd.Series(apo_list, index=df.index))
    df = df.assign(Trades=pd.Series(orders, index=df.index))
    df = df.assign(Position=pd.Series(positions, index=df.index))
    df = df.assign(PnL=pd.Series(pnls, index=df.index))
    return df


def volatility_mean_reversion_static_risk(
        prices,
        risk_limit_weekly_stop_loss,
        risk_limit_monthly_stop_loss,
        risk_limit_max_positions,
        risk_limit_max_positions_holding_time_days,
        risk_limit_max_trade_size,
        risk_limit_max_traded_volume,
        sma_time_periods=20,
        avg_std_dev=None,
        ema_time_period_fast=10,
        ema_time_period_slow=40,
        apo_value_for_buy_entry=-10,
        apo_value_for_sell_entry=10,
        min_price_move_from_last_trade=10,
        num_shares_per_trade=10,
        min_profit_to_close=10):
    """Return volatility adjusted mean reversion strategy with static risk.

    This function implements a mean reversion trading strategy that relies on
    the Absolute Price Oscillator (APO) trading signal. It incorporates risk
    management strategies using constant risk limits set to 150% of the maximum
    historical risk and performance limits. This buffer allows for the
    possibility of future trading scenarios that are different from historical
    trends.

    By default it uses 10 days for the fast EMA and 40 days for the slow EMA.
    It will perform buy trades when the APO signal value drops below -10 and
    perform sell trades when the APO signal value goes above +10. It will check
    that new trades are made at prices that are different from the last trade
    price to prevent over trading. Positions are closed when the APO signal
    value changes sign:
        * Close short positions when the APO goes negative, and
        * Close long positions when the APO goes positive.
    Positions are also closed if current open positions are profitable above a
    certain amount, regardless of the APO values. This is used to
    algorithmically lock profits and initiate more positions instead of relying
    on the trading signal value.

    :param Series prices: Price series.
    :param any risk_limit_weekly_stop_loss: Weekly stop loss risk limit.
    :param any risk_limit_monthly_stop_loss: Monthly stop loss risk limit.
    :param any risk_limit_max_positions: Maximum number of positions.
    :param any risk_limit_max_positions_holding_time_days: Maximum number of
        days that positions can be held.
    :param any risk_limit_max_trade_size: Maximum trade size risk limit.
    :param any risk_limit_max_traded_volume: Maximum traded volume risk limit.
    :param int sma_time_periods: Simple moving average look back period,
        default=20.
    :param avg_std_dev: Average standard deviation of prices SMA over
        look back period of sma_time_periods number of days. If this is not
        specified, it is calculated by the function, default=None.
    :param int ema_time_period_fast: Number of time periods for fast EMA,
        default=10.
    :param int ema_time_period_slow: Number of time periods for slow EMA,
        default=40.
    :param int apo_value_for_buy_entry: APO trading signal value below which to
        enter buy orders/long positions, default = -10.
    :param int apo_value_for_sell_entry: APO trading signal value above which to
        enter sell orders/short positions, default=10.
    :param int min_price_move_from_last_trade: Minimum price change since last
        trade before considering trading again. This prevents over trading at
        around the same prices, default=10.
    :param int num_shares_per_trade: Number of shares to buy/sell on every
        trade, default=10.
    :param int min_profit_to_close: Minimum open/unrealised profit at which to
         close and lock profits, default=10.
    :return: DataFrame containing the following columns:
        ClosePrice = price series provided as a parameter to the function.
        FastEMA = Fast exponential moving average.
        SlowEMA = Slow exponential moving average.
        APO = Absolute price oscillator.
        Trades = Buy/sell orders: buy=+1; sell=-1; no action=0.
        Positions = Long=+ve; short=-ve, flat/no position=0.
        PnL = Profit and loss.
    """
    # Variables for EMA calculation
    k_fast = 2 / (ema_time_period_fast + 1)  # fast EMA smoothing factor
    ema_fast = 0
    ema_fast_list = []  # calculated fast EMA values

    k_slow = 2 / (ema_time_period_slow + 1)  # slow EMA smoothing factor
    ema_slow = 0
    ema_slow_list = []  # calculated slow EMA values

    apo_list = []  # calculated absolute price oscillated signals

    # Variables for trading strategy trade, position and p&l management

    # Track buy/sell orders: buy=+1, sell=-1, no action=0
    orders = []
    # Track positions: long=+ve, short=-ve, flat/no position=0
    positions = []
    # Track total p&l
    pnls = []
    # Price at which last buy trade was made; used to prevent over trading
    last_buy_price = 0
    # Price at which last sell trade was made; used to prevent over trading
    last_sell_price = 0
    # Current position of the trading strategy
    position = 0
    # Sum of buy_trade_price and buy_trade_qty for every buy trade made since
    # last time being flat
    buy_sum_price_qty = 0
    # Summation of buy_trade_qty for every buy trade made since last time being
    # flat
    buy_sum_qty = 0
    # Sum of products of sell_trade_price and sell_trade_qty for every sell
    # trade made since last time being flat
    sell_sum_price_qty = 0
    # Sum of sell_trade_qty for every sell Trade made since last time being
    # flat
    sell_sum_qty = 0
    # Open/unrealised PnL marked to market
    open_pnl = 0
    # Closed/realised PnL so far
    closed_pnl = 0

    # Price history over sma_time_periods number of time periods for SMA and
    # standard deviation calculation used as a volatility measure
    price_history = []

    # Performance and risk limits
    risk_violated = False  # risk violation state tracking flag
    traded_volume = 0
    current_pos = 0
    current_pos_start = 0

    # Calculate average standard deviation of prices SMA if required
    if avg_std_dev is None:
        std_dev_list = standard_deviation(prices, time_period=sma_time_periods)
        avg_std_dev = stats.mean(std_dev_list)

    # Trading strategy main loop
    for close_price in prices:
        price_history.append(close_price)
        # Only track at most sma_time_periods number of prices
        if len(price_history) > sma_time_periods:
            del price_history[0]

        # Calculated SMA over sma_time_periods number of days
        sma = stats.mean(price_history)
        # Calculate variance over sma_time_periods number of days
        variance = 0  # variance is square of standard deviation
        for hist_price in price_history:
            # variance = variance + ((hist_price - sma) ** 2)
            variance += (hist_price - sma) ** 2

        stdev = math.sqrt(variance / len(price_history))
        stdev_factor = stdev / avg_std_dev
        if stdev_factor == 0:
            stdev_factor = 1

        # Calculate the fast and slow EMAs with smoothing factors adjusted for
        # volatility
        if ema_fast == 0:  # first observation
            ema_fast = close_price
            ema_slow = close_price
        else:
            ema_fast = (close_price - ema_fast) \
                       * k_fast * stdev_factor + ema_fast
            ema_slow = (close_price - ema_slow) \
                       * k_slow * stdev_factor + ema_slow

        ema_fast_list.append(ema_fast)
        ema_slow_list.append(ema_slow)

        # Calculate APO trading signal based on volatility adjusted EMAs
        apo = ema_fast - ema_slow
        apo_list.append(apo)

        # Ensure trade size is within maximum trade size risk limit
        # TODO: Verify this condition, it is checking two constants.
        # num_shares_per_trade never gets updated in the trading loop. Should
        # this be either position or traded_volume
        if num_shares_per_trade > risk_limit_max_trade_size:
            print('Risk violation: number of shares per trade',
                  num_shares_per_trade, '> risk limit max trade size',
                  risk_limit_max_trade_size)
            risk_violated = True

        # Check trading signal against trading parameters/thresholds and
        # positions to trade. This code uses dynamic thresholds based on
        # volatility for APO buy and sell entry thresholds. This makes the
        # strategy less aggressive in entering positions during periods of
        # higher volatility by increasing the threshold for entry by a factor
        # of volatility. Additionally, volatility is incorporated in the
        # expected profit threshold to lock in profit in a position by having
        # a dynamic threshold based on volatility.

        # Perform a sell trade at close_price on the following conditions:
        # 1. APO trading signal value is above sell entry threshold and the
        #    difference between last trade price and current price is different
        #    enough, or
        # 2. We are long (+ve position) and either APO trading signal value is
        #    at or above 0 or current position is profitable enough to lock
        #    profit.
        # 3. There are no risk limit violations.
        if not risk_violated and \
                ((apo > apo_value_for_sell_entry * stdev_factor and abs(
                    close_price - last_sell_price) >
                  min_price_move_from_last_trade * stdev_factor)
                 or
                 (position > 0 and (apo >= 0 or open_pnl >
                                    min_profit_to_close / stdev_factor))):
            orders.append(-1)  # mark the sell trade
            last_sell_price = close_price
            position -= num_shares_per_trade  # reduce position by size of trade
            sell_sum_price_qty += close_price * num_shares_per_trade
            sell_sum_qty += num_shares_per_trade
            traded_volume += num_shares_per_trade
            print('Sell', num_shares_per_trade, '@', close_price,
                  'Position:', position)

        # Perform a buy trade at close_price on the following conditions:
        # 1. APO trading signal value is below buy entry threshold and the
        #    difference between last trade price and current price is different
        #    enough, or
        # 2. We are short (-ve position) and either APO trading signal value is
        #    at or below 0 or current position is profitable enough to lock
        #    profit.
        # 3. There are no risk violations.
        elif not risk_violated and \
                ((apo < apo_value_for_buy_entry * stdev_factor and abs(
                    close_price - last_buy_price) >
                  min_price_move_from_last_trade * stdev_factor)
                 or
                 (position < 0 and (apo <= 0 or open_pnl >
                                    min_profit_to_close / stdev_factor))):
            orders.append(+1)  # mark the buy trade
            last_buy_price = close_price
            position += num_shares_per_trade  # increase position by trade size
            buy_sum_price_qty += close_price * num_shares_per_trade
            buy_sum_qty += num_shares_per_trade
            traded_volume += num_shares_per_trade
            print('Buy', num_shares_per_trade, '@', close_price,
                  'Position:', position)
        else:
            # No trade since none of the conditions were met to buy or sell
            orders.append(0)

        positions.append(position)

        # Check for any breaches of risk limits after any potential orders have
        # been sent out and trades have been made in this round, starting with
        # the maximum position holding time risk limit.

        # Flat and starting a new position
        if current_pos == 0:
            if position != 0:
                current_pos = position
                current_pos_start = len(positions)
        # Going from long position to flat or short position or
        # going from short position to flat or long position
        elif current_pos * position <= 0:
            current_pos = position
            position_holding_time = len(positions) - current_pos_start
            current_pos_start = len(positions)

            if position_holding_time > \
                    risk_limit_max_positions_holding_time_days:
                print('Risk Violation: position holding time',
                      position_holding_time,
                      '> risk limit max position holding time days',
                      risk_limit_max_positions_holding_time_days)
                risk_violated = True

        # Check new long/short position is within the maximum positions risk
        # limit
        if abs(position) > risk_limit_max_positions:
            print('Risk Violation: position', position,
                  '> risk limit max positions', risk_limit_max_positions)
            risk_violated = True

        # Check the updated traded volume doesn't violate the allocated maximum
        # traded volume risk limit
        if traded_volume > risk_limit_max_traded_volume:
            print('Risk Violation: traded volume', traded_volume,
                  '> risk limit max traded volume',
                  risk_limit_max_traded_volume)
            risk_violated = True

        # Update open/unrealised and closed/realised positions
        open_pnl = 0
        if position > 0:
            if sell_sum_qty > 0:
                # Long position and some sell trades have been made against it,
                # close that amount based on how much was sold against this
                # long position.
                open_pnl = abs(sell_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # would be if we closed at current price.
            open_pnl += abs(sell_sum_qty - position) * (
                    close_price - buy_sum_price_qty / buy_sum_qty)
        elif position < 0:
            if buy_sum_qty > 0:
                # Short position and some buy trades have been made against it,
                # close that amount based on how much was bought against this
                # short position.
                open_pnl = abs(buy_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # wold be if we closed at current price
            open_pnl += abs(buy_sum_qty - position) * (
                    sell_sum_price_qty / sell_sum_qty - close_price)
        else:
            # Flat, so update closed pnl and reset tracking variables for
            # positions and pnls
            closed_pnl += sell_sum_price_qty - buy_sum_price_qty
            buy_sum_price_qty = 0
            buy_sum_qty = 0
            sell_sum_price_qty = 0
            sell_sum_qty = 0
            last_buy_price = 0
            last_sell_price = 0

        print('OpenPnL:', open_pnl, 'ClosedPnL:', closed_pnl,
              'TotalPnL:', (open_pnl + closed_pnl))
        pnls.append(closed_pnl + open_pnl)

        # Check the new total PnL does not violate either the maximum allowed
        # weekly stop limit or maximum allowed monthly stop limit
        if len(pnls) > 5:
            weekly_loss = pnls[-1] - pnls[-6]

            if weekly_loss < risk_limit_weekly_stop_loss:
                print('Risk Violation: weekly loss', weekly_loss,
                      '< risk limit weekly stop loss',
                      risk_limit_weekly_stop_loss)
                risk_violated = True

        if len(pnls) > 20:
            monthly_loss = pnls[-1] - pnls[-21]

            if monthly_loss < risk_limit_monthly_stop_loss:
                print('Risk Violated: monthly loss', monthly_loss,
                      '< risk limit monthly stop loss',
                      risk_limit_monthly_stop_loss)
                risk_violated = True

    # Prepare DataFrame from the trading strategy results
    df = prices.to_frame(name='ClosePrice')
    df = df.assign(FastEMA=pd.Series(ema_fast_list, index=df.index))
    df = df.assign(SlowEMA=pd.Series(ema_slow_list, index=df.index))
    df = df.assign(APO=pd.Series(apo_list, index=df.index))
    df = df.assign(Trades=pd.Series(orders, index=df.index))
    df = df.assign(Position=pd.Series(positions, index=df.index))
    df = df.assign(PnL=pd.Series(pnls, index=df.index))
    return df


def volatility_mean_reversion_dynamic_risk(
        prices,
        risk_limit_weekly_stop_loss,
        increment_risk_limit_weekly_stop_loss,
        risk_limit_monthly_stop_loss,
        increment_risk_limit_monthly_stop_loss,
        risk_limit_max_positions,
        increment_risk_limit_max_positions,
        risk_limit_max_positions_holding_time_days,
        risk_limit_max_trade_size,
        increment_risk_limit_max_trade_size,
        sma_time_periods=20,
        avg_std_dev=None,
        ema_time_period_fast=10,
        ema_time_period_slow=40,
        apo_value_for_buy_entry=-10,
        apo_value_for_sell_entry=10,
        min_price_move_from_last_trade=10,
        min_num_shares_per_trade=1,
        max_num_shares_per_trade=50,
        increment_num_shares_per_trade=2,
        min_profit_to_close=10):
    """Return volatility adjusted mean reversion strategy with dynamic risk.

    This function implements a mean reversion trading strategy that relies on
    the Absolute Price Oscillator (APO) trading signal. It incorporates risk
    management strategies using dynamically adjusted risk limits between
    predefined limits.

    By default it uses 10 days for the fast EMA and 40 days for the slow EMA.
    It will perform buy trades when the APO signal value drops below -10 and
    perform sell trades when the APO signal value goes above +10. It will check
    that new trades are made at prices that are different from the last trade
    price to prevent over trading. Positions are closed when the APO signal
    value changes sign:
        * Close short positions when the APO goes negative, and
        * Close long positions when the APO goes positive.
    Positions are also closed if current open positions are profitable above a
    certain amount, regardless of the APO values. This is used to
    algorithmically lock profits and initiate more positions instead of relying
    on the trading signal value.

    :param Series prices: Price series.
    :param any risk_limit_weekly_stop_loss: Weekly stop loss risk limit.
    :param any increment_risk_limit_weekly_stop_loss: Weekly stop loss risk
        limit increment.
    :param any risk_limit_monthly_stop_loss: Monthly stop loss risk limit.
    :param any increment_risk_limit_monthly_stop_loss: Monthly stop loss risk
        limit increment.
    :param any risk_limit_max_positions: Maximum number of positions.
    :param any increment_risk_limit_max_positions: Maximum number of positions
        risk limit increment.
    :param any risk_limit_max_positions_holding_time_days: Maximum number of
        days that positions can be held.
    :param any risk_limit_max_trade_size: Maximum trade size risk limit.
    :param any increment_risk_limit_max_trade_size: Maximum trade size risk
        limit increment.
    :param int sma_time_periods: Simple moving average look back period,
        default=20.
    :param avg_std_dev: Average standard deviation of prices SMA over
        look back period of sma_time_periods number of days. If this is not
        specified, it is calculated by the function, default=None.
    :param int ema_time_period_fast: Number of time periods for fast EMA,
        default=10.
    :param int ema_time_period_slow: Number of time periods for slow EMA,
        default=40.
    :param int apo_value_for_buy_entry: APO trading signal value below which to
        enter buy orders/long positions, default = -10.
    :param int apo_value_for_sell_entry: APO trading signal value above which to
        enter sell orders/short positions, default=10.
    :param int min_price_move_from_last_trade: Minimum price change since last
        trade before considering trading again. This prevents over trading at
        around the same prices, default=10.
    :param int min_num_shares_per_trade: Minimum number of shares to buy/sell
        on every trade, default=1.
    :param int max_num_shares_per_trade: Maximum number of shares to buy/sell
        on every trade, default=50.
    :param int increment_num_shares_per_trade: Increment in number of shares
        to buy/sell on every trade, default=2
    :param int min_profit_to_close: Minimum open/unrealised profit at which to
         close and lock profits, default=10.
    :return: DataFrame containing the following columns:
        ClosePrice = price series provided as a parameter to the function.
        FastEMA = Fast exponential moving average.
        SlowEMA = Slow exponential moving average.
        APO = Absolute price oscillator.
        Trades = Buy/sell orders: buy=+1; sell=-1; no action=0.
        Positions = Long=+ve; short=-ve, flat/no position=0.
        PnL = Profit and loss.
        NumShares = Number of shares history.
        MaxTradeSize = Maximum trade size history.
        AbsPosition = History of absolute positions.
        MaxPosition = History of maximum positions.
    """
    # Variables for EMA calculation
    k_fast = 2 / (ema_time_period_fast + 1)  # fast EMA smoothing factor
    ema_fast = 0
    ema_fast_list = []  # calculated fast EMA values

    k_slow = 2 / (ema_time_period_slow + 1)  # slow EMA smoothing factor
    ema_slow = 0
    ema_slow_list = []  # calculated slow EMA values

    apo_list = []  # calculated absolute price oscillated signals

    # Variables for trading strategy trade, position and p&l management

    # Track buy/sell orders: buy=+1, sell=-1, no action=0
    orders = []
    # Track positions: long=+ve, short=-ve, flat/no position=0
    positions = []
    # Track total p&l
    pnls = []
    # Price at which last buy trade was made; used to prevent over trading
    last_buy_price = 0
    # Price at which last sell trade was made; used to prevent over trading
    last_sell_price = 0
    # Current position of the trading strategy
    position = 0
    # Sum of buy_trade_price and buy_trade_qty for every buy trade made since
    # last time being flat
    buy_sum_price_qty = 0
    # Summation of buy_trade_qty for every buy trade made since last time being
    # flat
    buy_sum_qty = 0
    # Sum of products of sell_trade_price and sell_trade_qty for every sell
    # trade made since last time being flat
    sell_sum_price_qty = 0
    # Sum of sell_trade_qty for every sell Trade made since last time being
    # flat
    sell_sum_qty = 0
    # Open/unrealised PnL marked to market
    open_pnl = 0
    # Closed/realised PnL so far
    closed_pnl = 0
    # Beginning number of shares to buy/sell on every trade
    num_shares_per_trade = min_num_shares_per_trade
    num_shares_history = []  # history of number of shares
    abs_position_history = []  # history of absolute position

    # Price history over sma_time_periods number of time periods for SMA and
    # standard deviation calculation used as a volatility measure
    price_history = []

    # Performance and risk limits
    risk_violated = False  # risk violation state tracking flag
    traded_volume = 0
    current_pos = 0
    current_pos_start = 0
    max_position_history = []  # history of maximum positions
    max_trade_size_history = []  # history of maximum trade size
    last_risk_change_index = 0

    # Calculate average standard deviation of prices SMA if required
    if avg_std_dev is None:
        std_dev_list = standard_deviation(prices, time_period=sma_time_periods)
        avg_std_dev = stats.mean(std_dev_list)

    # Trading strategy main loop
    for close_price in prices:
        price_history.append(close_price)
        # Only track at most sma_time_periods number of prices
        if len(price_history) > sma_time_periods:
            del price_history[0]

        # Calculated SMA over sma_time_periods number of days
        sma = stats.mean(price_history)
        # Calculate variance over sma_time_periods number of days
        variance = 0  # variance is square of standard deviation
        for hist_price in price_history:
            # variance = variance + ((hist_price - sma) ** 2)
            variance += (hist_price - sma) ** 2

        stdev = math.sqrt(variance / len(price_history))
        stdev_factor = stdev / avg_std_dev
        if stdev_factor == 0:
            stdev_factor = 1

        # Calculate the fast and slow EMAs with smoothing factors adjusted for
        # volatility
        if ema_fast == 0:  # first observation
            ema_fast = close_price
            ema_slow = close_price
        else:
            ema_fast = (close_price - ema_fast) \
                       * k_fast * stdev_factor + ema_fast
            ema_slow = (close_price - ema_slow) \
                       * k_slow * stdev_factor + ema_slow

        ema_fast_list.append(ema_fast)
        ema_slow_list.append(ema_slow)

        # Calculate APO trading signal based on volatility adjusted EMAs
        apo = ema_fast - ema_slow
        apo_list.append(apo)

        # Ensure trade size is within maximum trade size risk limit
        # TODO: Verify this condition, it is checking two constants.
        # num_shares_per_trade never gets updated in the trading loop. Should
        # this be either position or traded_volume
        if num_shares_per_trade > risk_limit_max_trade_size:
            print('Risk violation: number of shares per trade',
                  num_shares_per_trade, '> risk limit max trade size',
                  risk_limit_max_trade_size)
            risk_violated = True

        # Check trading signal against trading parameters/thresholds and
        # positions to trade. This code uses dynamic thresholds based on
        # volatility for APO buy and sell entry thresholds. This makes the
        # strategy less aggressive in entering positions during periods of
        # higher volatility by increasing the threshold for entry by a factor
        # of volatility. Additionally, volatility is incorporated in the
        # expected profit threshold to lock in profit in a position by having
        # a dynamic threshold based on volatility.

        # Perform a sell trade at close_price on the following conditions:
        # 1. APO trading signal value is above sell entry threshold and the
        #    difference between last trade price and current price is different
        #    enough, or
        # 2. We are long (+ve position) and either APO trading signal value is
        #    at or above 0 or current position is profitable enough to lock
        #    profit.
        # 3. There are no risk limit violations.
        if not risk_violated and \
                ((apo > apo_value_for_sell_entry * stdev_factor and abs(
                    close_price - last_sell_price) >
                  min_price_move_from_last_trade * stdev_factor)
                 or
                 (position > 0 and (apo >= 0 or open_pnl >
                                    min_profit_to_close / stdev_factor))):
            orders.append(-1)  # mark the sell trade
            last_sell_price = close_price
            if position == 0:  # opening a new entry position
                # reduce position by size of trade
                position -= num_shares_per_trade
                sell_sum_price_qty += close_price * num_shares_per_trade
                sell_sum_qty += num_shares_per_trade
                traded_volume += num_shares_per_trade
                print('Sell', num_shares_per_trade, '@', close_price,
                      'Position:', position)
            else:  # closing an existing position
                sell_sum_price_qty += close_price * abs(position)
                sell_sum_qty += abs(position)
                traded_volume += abs(position)
                print('Sell', abs(position), '@', close_price,
                      'Position:', position)
                position = 0  # close position

        # Perform a buy trade at close_price on the following conditions:
        # 1. APO trading signal value is below buy entry threshold and the
        #    difference between last trade price and current price is different
        #    enough, or
        # 2. We are short (-ve position) and either APO trading signal value is
        #    at or below 0 or current position is profitable enough to lock
        #    profit.
        # 3. There are no risk violations.
        elif not risk_violated and \
                ((apo < apo_value_for_buy_entry * stdev_factor and abs(
                    close_price - last_buy_price) >
                  min_price_move_from_last_trade * stdev_factor)
                 or
                 (position < 0 and (apo <= 0 or open_pnl >
                                    min_profit_to_close / stdev_factor))):
            orders.append(+1)  # mark the buy trade
            last_buy_price = close_price
            if position == 0:  # opening a new entry position
                # increase position by trade size
                position += num_shares_per_trade
                buy_sum_price_qty += close_price * num_shares_per_trade
                buy_sum_qty += num_shares_per_trade
                traded_volume += num_shares_per_trade
                print('Buy', num_shares_per_trade, '@', close_price,
                      'Position:', position)
            else:  # closing an existing position
                buy_sum_price_qty += close_price * abs(position)
                buy_sum_qty += abs(position)
                traded_volume += abs(position)
                print('Buy', abs(position), '@', close_price,
                      'Position:', position)
                position = 0  # close position
        else:
            # No trade since none of the conditions were met to buy or sell
            orders.append(0)

        positions.append(position)

        # Check for any breaches of risk limits after any potential orders have
        # been sent out and trades have been made in this round, starting with
        # the maximum position holding time risk limit.

        # Flat and starting a new position
        if current_pos == 0:
            if position != 0:
                current_pos = position
                current_pos_start = len(positions)
        # Going from long position to flat or short position or
        # going from short position to flat or long position
        elif current_pos * position <= 0:
            current_pos = position
            position_holding_time = len(positions) - current_pos_start
            current_pos_start = len(positions)

            if position_holding_time > \
                    risk_limit_max_positions_holding_time_days:
                print('Risk Violation: position holding time',
                      position_holding_time,
                      '> risk limit max position holding time days',
                      risk_limit_max_positions_holding_time_days)
                risk_violated = True

        # Check new long/short position is within the maximum positions risk
        # limit
        if abs(position) > risk_limit_max_positions:
            print('Risk Violation: position', position,
                  '> risk limit max positions', risk_limit_max_positions)
            risk_violated = True

        # Update open/unrealised and closed/realised positions
        open_pnl = 0
        if position > 0:
            if sell_sum_qty > 0:
                # Long position and some sell trades have been made against it,
                # close that amount based on how much was sold against this
                # long position.
                open_pnl = abs(sell_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # would be if we closed at current price.
            open_pnl += abs(sell_sum_qty - position) * (
                    close_price - buy_sum_price_qty / buy_sum_qty)
        elif position < 0:
            if buy_sum_qty > 0:
                # Short position and some buy trades have been made against it,
                # close that amount based on how much was bought against this
                # short position.
                open_pnl = abs(buy_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # wold be if we closed at current price
            open_pnl += abs(buy_sum_qty - position) * (
                    sell_sum_price_qty / sell_sum_qty - close_price)
        else:
            # Flat, so update closed pnl and reset tracking variables for
            # positions and pnls
            closed_pnl += sell_sum_price_qty - buy_sum_price_qty
            buy_sum_price_qty = 0
            buy_sum_qty = 0
            sell_sum_price_qty = 0
            sell_sum_qty = 0
            last_buy_price = 0
            last_sell_price = 0

        print('OpenPnL:', open_pnl, 'ClosedPnL:', closed_pnl,
              'TotalPnL:', (open_pnl + closed_pnl))
        pnls.append(closed_pnl + open_pnl)

        # Analyse monthly performance and adjust risk limits up/down
        if len(pnls) > 20:
            monthly_pnls = pnls[-1] - pnls[-20]

            if len(pnls) - last_risk_change_index > 20:
                if monthly_pnls > 0:
                    num_shares_per_trade += increment_num_shares_per_trade
                    if num_shares_per_trade <= max_num_shares_per_trade:
                        print('Increasing trade size limit and risk')
                        risk_limit_weekly_stop_loss += \
                            increment_risk_limit_weekly_stop_loss
                        risk_limit_monthly_stop_loss += \
                            increment_risk_limit_monthly_stop_loss
                        risk_limit_max_positions += \
                            increment_risk_limit_max_positions
                        risk_limit_max_trade_size += \
                            increment_risk_limit_max_trade_size
                    else:
                        num_shares_per_trade = max_num_shares_per_trade
                elif monthly_pnls < 0:
                    num_shares_per_trade -= increment_num_shares_per_trade
                    if num_shares_per_trade >= min_num_shares_per_trade:
                        print('Decreasing trade size limit and risk')
                        risk_limit_weekly_stop_loss -= \
                            increment_risk_limit_weekly_stop_loss
                        risk_limit_monthly_stop_loss -= \
                            increment_risk_limit_monthly_stop_loss
                        risk_limit_max_positions -= \
                            increment_risk_limit_max_positions
                        risk_limit_max_trade_size -= \
                            increment_risk_limit_max_trade_size
                    else:
                        num_shares_per_trade = min_num_shares_per_trade

                last_risk_change_index = len(pnls)

        # Track trade sizes/positions and risk limits as they evolve over time
        num_shares_history.append(num_shares_per_trade)
        abs_position_history.append(abs(position))
        max_trade_size_history.append(risk_limit_max_trade_size)
        max_position_history.append(risk_limit_max_positions)

        # Check the new total PnL does not violate either the maximum allowed
        # weekly stop limit or maximum allowed monthly stop limit
        if len(pnls) > 5:
            weekly_loss = pnls[-1] - pnls[-6]

            if weekly_loss < risk_limit_weekly_stop_loss:
                print('Risk Violation: weekly loss', weekly_loss,
                      '< risk limit weekly stop loss',
                      risk_limit_weekly_stop_loss)
                risk_violated = True

        if len(pnls) > 20:
            monthly_loss = pnls[-1] - pnls[-21]

            if monthly_loss < risk_limit_monthly_stop_loss:
                print('Risk Violated: monthly loss', monthly_loss,
                      '< risk limit monthly stop loss',
                      risk_limit_monthly_stop_loss)
                risk_violated = True

    # Prepare DataFrame from the trading strategy results
    df = prices.to_frame(name='ClosePrice')
    df = df.assign(FastEMA=pd.Series(ema_fast_list, index=df.index))
    df = df.assign(SlowEMA=pd.Series(ema_slow_list, index=df.index))
    df = df.assign(APO=pd.Series(apo_list, index=df.index))
    df = df.assign(Trades=pd.Series(orders, index=df.index))
    df = df.assign(Position=pd.Series(positions, index=df.index))
    df = df.assign(PnL=pd.Series(pnls, index=df.index))
    df = df.assign(NumShares=pd.Series(num_shares_history, index=df.index))
    df = df.assign(
        MaxTradeSize=pd.Series(max_trade_size_history, index=df.index))
    df = df.assign(AbsPosition=pd.Series(abs_position_history, index=df.index))
    df = df.assign(MaxPosition=pd.Series(max_position_history, index=df.index))
    return df


def basic_trend_following(prices, ema_time_period_fast=10,
                          ema_time_period_slow=40, apo_value_for_buy_entry=10,
                          apo_value_for_sell_entry=-10,
                          min_price_move_from_last_trade=10,
                          num_shares_per_trade=10, min_profit_to_close=10):
    """Return trend following trading strategy results based on APO.

    This function implements a trend following trading strategy that relies on
    the Absolute Price Oscillator (APO) trading signal. By default it uses
    10 days for the fast EMA and 40 days for the slow EMA. It will perform
    buy trades when the APO signal value goes above +10 and perform sell trades
    when the APO signal value drops below -10. It will check that new trades are
    made at prices that are different from the last trade price to prevent over
    trading. Positions are closed when the APO signal value changes sign:
        * Close short positions when the APO goes negative, and
        * Close long positions when the APO goes positive.
    Positions are also closed if current open positions are profitable above a
    certain amount, regardless of the APO values. This is used to
    algorithmically lock profits and initiate more positions instead of relying
    on the trading signal value.

    :param Series prices: Price series.
    :param int ema_time_period_fast: Number of time periods for fast EMA,
        default=10.
    :param int ema_time_period_slow: Number of time periods for slow EMA,
        default=40.
    :param int apo_value_for_buy_entry: APO trading signal value above which to
        enter buy orders/long positions, default=10.
    :param int apo_value_for_sell_entry: APO trading signal value below which to
        enter sell orders/short positions, default=-10.
    :param int min_price_move_from_last_trade: Minimum price change since last
        trade before considering trading again. This prevents over trading at
        around the same prices, default=10.
    :param int num_shares_per_trade: Number of shares to buy/sell on every
        trade, default=10.
    :param int min_profit_to_close: Minimum open/unrealised profit at which to
         close and lock profits, default=10.
    :return: DataFrame containing the following columns:
        ClosePrice = price series provided as a parameter to the function.
        FastEMA = Fast exponential moving average.
        SlowEMA = Slow exponential moving average.
        APO = Absolute price oscillator.
        Trades = Buy/sell orders: buy=+1; sell=-1; no action=0.
        Positions = Long=+ve; short=-ve, flat/no position=0.
        PnL = Profit and loss.
    """
    # Variables for trading strategy trade, position and p&l management

    # Track buy/sell orders: buy=+1, sell=-1, no action=0
    orders = []
    # Track positions: long=+ve, short=-ve, flat/no position=0
    positions = []
    # Track total p&l
    pnls = []
    # Price at which last buy trade was made; used to prevent over trading
    last_buy_price = 0
    # Price at which last sell trade was made; used to prevent over trading
    last_sell_price = 0
    # Current position of the trading strategy
    position = 0
    # Sum of buy_trade_price and buy_trade_qty for every buy trade made since
    # last time being flat
    buy_sum_price_qty = 0
    # Summation of buy_trade_qty for every buy trade made since last time being
    # flat
    buy_sum_qty = 0
    # Sum of products of sell_trade_price and sell_trade_qty for every sell
    # trade made since last time being flat
    sell_sum_price_qty = 0
    # Sum of sell_trade_qty for every sell Trade made since last time being
    # flat
    sell_sum_qty = 0
    # Open/unrealised PnL marked to market
    open_pnl = 0
    # Closed/realised PnL so far
    closed_pnl = 0

    # Trading strategy

    # Calculate fast and slow EMA and APO on close price
    apo_df = absolute_price_oscillator(prices,
                                       time_period_fast=ema_time_period_fast,
                                       time_period_slow=ema_time_period_slow)
    ema_fast_values = apo_df.loc[:, 'ema_fast'].tolist()
    ema_slow_values = apo_df.loc[:, 'ema_slow'].tolist()
    apo_values = apo_df.loc[:, 'apo'].tolist()

    # Trading strategy main loop
    for close_price, apo in zip(prices, apo_values):
        # Check trading signal against trading parameters/thresholds and
        # positions to trade

        # Perform a sell trade at close_price on the following conditions:
        # 1. APO trading signal value is below sell entry threshold and the
        #    difference between last trade price and current price is different
        #    enough, or
        # 2. We are long (+ve position) and either APO trading signal value is
        #    at or below 0 or current position is profitable enough to lock
        #    profit.
        if ((apo < apo_value_for_sell_entry and abs(
                close_price - last_sell_price) > min_price_move_from_last_trade)
                or
                (position > 0 and (apo <= 0 or
                                   open_pnl > min_profit_to_close))):
            orders.append(-1)  # mark the sell trade
            last_sell_price = close_price
            position -= num_shares_per_trade  # reduce position by size of trade
            sell_sum_price_qty += close_price * num_shares_per_trade
            sell_sum_qty += num_shares_per_trade
            print('Sell ', num_shares_per_trade, ' @ ', close_price,
                  'Position: ', position)

        # Perform a buy trade at close_price on the following conditions:
        # 1. APO trading signal value is above buy entry threshold and the
        #    difference between last trade price and current price is different
        #    enough, or
        # 2. We are short (-ve position) and either APO trading signal value is
        #    at or above 0 or current position is profitable enough to lock
        #    profit.
        elif ((apo > apo_value_for_buy_entry and abs(
                close_price - last_buy_price) > min_price_move_from_last_trade)
              or
              (position < 0 and (apo >= 0 or open_pnl > min_profit_to_close))):
            orders.append(+1)  # mark the buy trade
            last_buy_price = close_price
            position += num_shares_per_trade  # increase position by trade size
            buy_sum_price_qty += close_price * num_shares_per_trade
            buy_sum_qty += num_shares_per_trade
            print('Buy ', num_shares_per_trade, ' @ ', close_price,
                  'Position: ', position)
        else:
            # No trade since none of the conditions were met to buy or sell
            orders.append(0)

        positions.append(position)

        # Update open/unrealised and closed/realised positions
        open_pnl = 0
        if position > 0:
            if sell_sum_qty > 0:
                # Long position and some sell trades have been made against it,
                # close that amount based on how much was sold against this
                # long position.
                open_pnl = abs(sell_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # would be if we closed at current price.
            open_pnl += abs(sell_sum_qty - position) * (
                    close_price - buy_sum_price_qty / buy_sum_qty)
        elif position < 0:
            if buy_sum_qty > 0:
                # Short position and some buy trades have been made against it,
                # close that amount based on how much was bought against this
                # short position.
                open_pnl = abs(buy_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # wold be if we closed at current price
            open_pnl += abs(buy_sum_qty - position) * (
                    sell_sum_price_qty / sell_sum_qty - close_price)
        else:
            # Flat, so update closed pnl and reset tracking variables for
            # positions and pnls
            closed_pnl += sell_sum_price_qty - buy_sum_price_qty
            buy_sum_price_qty = 0
            buy_sum_qty = 0
            sell_sum_price_qty = 0
            sell_sum_qty = 0
            last_buy_price = 0
            last_sell_price = 0

        print('OpenPnL: ', open_pnl, ' ClosedPnL: ', closed_pnl,
              ' TotalPnL', (open_pnl + closed_pnl))
        pnls.append(closed_pnl + open_pnl)

    # Prepare DataFrame from the trading strategy results
    df = prices.to_frame(name='ClosePrice')
    df = df.assign(FastEMA=pd.Series(ema_fast_values, index=df.index))
    df = df.assign(SlowEMA=pd.Series(ema_slow_values, index=df.index))
    df = df.assign(APO=pd.Series(apo_values, index=df.index))
    df = df.assign(Trades=pd.Series(orders, index=df.index))
    df = df.assign(Position=pd.Series(positions, index=df.index))
    df = df.assign(PnL=pd.Series(pnls, index=df.index))
    return df


def volatility_trend_following(prices, sma_time_periods=20, avg_std_dev=None,
                               ema_time_period_fast=10, ema_time_period_slow=40,
                               apo_value_for_buy_entry=10,
                               apo_value_for_sell_entry=-10,
                               min_price_move_from_last_trade=10,
                               num_shares_per_trade=10, min_profit_to_close=10):
    """Return trend following trading strategy based on volatility adjusted APO.

    This function uses the standard deviation as a volatility measure to
    adjust the  number of days used in the fast and slow EAM to produce a
    volatility adjusted APO entry signal. By default it uses 10 days for the
    fast EMA and 40 days for the slow EMA. It will perform buy trades when the
    APO signal value goes above +10 and perform sell trades when the APO
    signal value drops below -10. It will check that new trades are made at
    prices that are different from the last trade price to prevent over trading.
    Positions are closed when the APO signal value changes sign:
        * Close short positions when the APO goes negative, and
        * Close long positions when the APO goes positive.
    Positions are also closed if current open positions are profitable above a
    certain amount, regardless of the APO values. This is used to
    algorithmically lock profits and initiate more positions instead of relying
    on the trading signal value.

    :param Series prices: Price series.
    :param int sma_time_periods: Simple moving average look back period,
        default=20.
    :param avg_std_dev: Average standard deviation of prices SMA over
        look back period of sma_time_periods number of days. If this is not
        specified, it is calculated by the function, default=None.
    :param int ema_time_period_fast: Number of time periods for fast EMA,
        default=10.
    :param int ema_time_period_slow: Number of time periods for slow EMA,
        default=40.
    :param int apo_value_for_buy_entry: APO trading signal value above which to
        enter buy orders/long positions, default=10.
    :param int apo_value_for_sell_entry: APO trading signal value below which to
        enter sell orders/short positions, default=-10.
    :param int min_price_move_from_last_trade: Minimum price change since last
        trade before considering trading again. This prevents over trading at
        around the same prices, default=10.
    :param int num_shares_per_trade: Number of shares to buy/sell on every
        trade, default=10.
    :param int min_profit_to_close: Minimum open/unrealised profit at which to
         close and lock profits, default=10.
    :return: DataFrame containing the following columns:
        ClosePrice = price series provided as a parameter to the function.
        FastEMA = Fast exponential moving average.
        SlowEMA = Slow exponential moving average.
        APO = Absolute price oscillator.
        Trades = Buy/sell orders: buy=+1; sell=-1; no action=0.
        Positions = Long=+ve; short=-ve, flat/no position=0.
        PnL = Profit and loss.
    """
    # Variables for EMA calculation
    k_fast = 2 / (ema_time_period_fast + 1)  # fast EMA smoothing factor
    ema_fast = 0
    ema_fast_list = []  # calculated fast EMA values

    k_slow = 2 / (ema_time_period_slow + 1)  # slow EMA smoothing factor
    ema_slow = 0
    ema_slow_list = []  # calculated slow EMA values

    apo_list = []  # calculated absolute price oscillated signals

    # Variables for trading strategy trade, position and p&l management

    # Track buy/sell orders: buy=+1, sell=-1, no action=0
    orders = []
    # Track positions: long=+ve, short=-ve, flat/no position=0
    positions = []
    # Track total p&l
    pnls = []
    # Price at which last buy trade was made; used to prevent over trading
    last_buy_price = 0
    # Price at which last sell trade was made; used to prevent over trading
    last_sell_price = 0
    # Current position of the trading strategy
    position = 0
    # Sum of products of buy_trade_price and buy_trade_qty for every buy trade
    # made since last time being flat
    buy_sum_price_qty = 0
    # Summation of buy_trade_qty for every buy trade made since last time being
    # flat
    buy_sum_qty = 0
    # Sum of products of sell_trade_price and sell_trade_qty for every sell
    # trade made since last time being flat
    sell_sum_price_qty = 0
    # Sum of sell_trade_qty for every sell Trade made since last time being
    # flat
    sell_sum_qty = 0
    # Open/unrealised PnL marked to market
    open_pnl = 0
    # Closed/realised PnL so far
    closed_pnl = 0

    # Price history over sma_time_periods number of time periods for SMA and
    # standard deviation calculation used as a volatility measure
    price_history = []

    # Calculate average standard deviation of prices SMA if required
    if avg_std_dev is None:
        std_dev_list = standard_deviation(prices, time_period=sma_time_periods)
        avg_std_dev = stats.mean(std_dev_list)

    # Trading strategy main loop
    for close_price in prices:
        price_history.append(close_price)
        # Only track at most sma_time_periods number of prices
        if len(price_history) > sma_time_periods:
            del price_history[0]

        # Calculated SMA over sma_time_periods number of days
        sma = stats.mean(price_history)
        # Calculate variance over sma_time_periods number of days
        variance = 0  # variance is square of standard deviation
        for hist_price in price_history:
            variance = variance + ((hist_price - sma) ** 2)

        stdev = math.sqrt(variance / len(price_history))
        stdev_factor = stdev / avg_std_dev
        if stdev_factor == 0:
            stdev_factor = 1

        # Calculate the fast and slow EMAs with smoothing factors adjusted for
        # volatility
        if ema_fast == 0:  # first observation
            ema_fast = close_price
            ema_slow = close_price
        else:
            ema_fast = (close_price - ema_fast) \
                       * k_fast * stdev_factor + ema_fast
            ema_slow = (close_price - ema_slow) \
                       * k_slow * stdev_factor + ema_slow

        ema_fast_list.append(ema_fast)
        ema_slow_list.append(ema_slow)

        # Calculate APO trading signal based on volatility adjusted EMAs
        apo = ema_fast - ema_slow
        apo_list.append(apo)

        # Check trading signal against trading parameters/thresholds and
        # positions to trade. This code uses dynamic thresholds based on
        # volatility for APO buy and sell entry thresholds. This makes the
        # strategy less aggressive in entering positions during periods of
        # higher volatility by increasing the threshold for entry by a factor
        # of volatility. Additionally, volatility is incorporated in the
        # expected profit threshold to lock in profit in a position by having
        # a dynamic threshold based on volatility.

        # Perform a sell trade at close_price on the following conditions:
        # 1. APO trading signal value is below sell entry threshold and the
        #    difference between last trade price and current price is different
        #    enough, or
        # 2. We are long (+ve position) and either APO trading signal value is
        #    at or above 0 or current position is profitable enough to lock
        #    profit.
        if ((apo < apo_value_for_sell_entry / stdev_factor and abs(
                close_price - last_sell_price) >
             min_price_move_from_last_trade * stdev_factor)
                or
                (position > 0 and (apo <= 0 or open_pnl >
                                   min_profit_to_close / stdev_factor))):
            orders.append(-1)  # mark the sell trade
            last_sell_price = close_price
            position -= num_shares_per_trade  # reduce position by size of trade
            sell_sum_price_qty += close_price * num_shares_per_trade
            sell_sum_qty += num_shares_per_trade
            print('Sell ', num_shares_per_trade, ' @ ', close_price,
                  'Position: ', position)

        # Perform a buy trade at close_price on the following conditions:
        # 1. APO trading signal value is above buy entry threshold and the
        #    difference between last trade price and current price is different
        #    enough, or
        # 2. We are short (-ve position) and either APO trading signal value is
        #    at or below 0 or current position is profitable enough to lock
        #    profit.
        elif ((apo > apo_value_for_buy_entry / stdev_factor and abs(
                close_price - last_buy_price) >
               min_price_move_from_last_trade * stdev_factor)
              or
              (position < 0 and (apo >= 0 or open_pnl >
                                 min_profit_to_close / stdev_factor))):
            orders.append(+1)  # mark the buy trade
            last_buy_price = close_price
            position += num_shares_per_trade  # increase position by trade size
            buy_sum_price_qty += close_price * num_shares_per_trade
            buy_sum_qty += num_shares_per_trade
            print('Buy ', num_shares_per_trade, ' @ ', close_price,
                  'Position: ', position)
        else:
            # No trade since none of the conditions were met to buy or sell
            orders.append(0)

        positions.append(position)

        # Update open/unrealised and closed/realised positions
        open_pnl = 0
        if position > 0:
            if sell_sum_qty > 0:
                # Long position and some sell trades have been made against it,
                # close that amount based on how much was sold against this
                # long position.
                open_pnl = abs(sell_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # would be if we closed at current price.
            open_pnl += abs(sell_sum_qty - position) * (
                    close_price - buy_sum_price_qty / buy_sum_qty)
        elif position < 0:
            if buy_sum_qty > 0:
                # Short position and some buy trades have been made against it,
                # close that amount based on how much was bought against this
                # short position.
                open_pnl = abs(buy_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark remaining position to market i.e. pnl would be what it
            # wold be if we closed at current price
            open_pnl += abs(buy_sum_qty - position) * (
                    sell_sum_price_qty / sell_sum_qty - close_price)
        else:
            # Flat, so update closed pnl and reset tracking variables for
            # positions and pnls
            closed_pnl += sell_sum_price_qty - buy_sum_price_qty
            buy_sum_price_qty = 0
            buy_sum_qty = 0
            sell_sum_price_qty = 0
            sell_sum_qty = 0
            last_buy_price = 0
            last_sell_price = 0

        print('OpenPnL: ', open_pnl, ' ClosedPnL: ', closed_pnl,
              ' TotalPnL', (open_pnl + closed_pnl))
        pnls.append(closed_pnl + open_pnl)

    # Prepare DataFrame from the trading strategy results
    df = prices.to_frame(name='ClosePrice')
    df = df.assign(FastEMA=pd.Series(ema_fast_list, index=df.index))
    df = df.assign(SlowEMA=pd.Series(ema_slow_list, index=df.index))
    df = df.assign(APO=pd.Series(apo_list, index=df.index))
    df = df.assign(Trades=pd.Series(orders, index=df.index))
    df = df.assign(Position=pd.Series(positions, index=df.index))
    df = df.assign(PnL=pd.Series(pnls, index=df.index))
    return df


def currency_stat_arb(symbols_data, trading_instrument, sma_time_periods=20,
                      price_dev_num_prices=200,
                      stat_arb_value_for_buy_entry=0.01,
                      stat_arb_value_for_sell_entry=-0.01,
                      min_price_move_from_last_trade=0.01,
                      num_shares_per_trade=1000000,
                      min_profit_to_close=10):
    """Statistical arbitrage of currency pairs.

    :param dict symbols_data: Dictionary of fireign currency historical data.
    :param str trading_instrument: Trading instrument dictionary key.
    :param int sma_time_periods: Simple moving average look back period,
        default=20.
    :param int price_dev_num_prices: Look back period of close price deviations
        from SMA, default=200.
    :param float stat_arb_value_for_buy_entry: Statistical arbitrage trading 
        signal value above which to enter buy orders/long positions, 
        default=0.01.
    :param float stat_arb_value_for_sell_entry: Statistical arbitrage trading 
        signal value below which to enter sell orders/short positions, 
        default=-0.01.
    :param float min_price_move_from_last_trade: Minimum price change since last
        trade before considering trading again. This prevents over trading at
        around the same prices, default=0.01.
    :param int num_shares_per_trade: Number of currencies to buy/sell on every
        trade, default=1000000.
    :param int min_profit_to_close: Minimum open/unrealised profit at which to
         close and lock profits, default=10.
    :return: DataFrame containing the following columns:
        ClosePrice = price series provided as a parameter to the function.
        FastEMA = Fast exponential moving average.
        SlowEMA = Slow exponential moving average.
        APO = Absolute price oscillator.
        Trades = Buy/sell orders: buy=+1; sell=-1; no action=0.
        Positions = Long=+ve; short=-ve, flat/no position=0.
        PnL = Profit and loss.
    """
    price_history = {}  # history of prices
    price_deviation_from_sma = {}  # history of close price deviation from SMA
    # Length of data for trading instrument (number of days)
    num_days = len(symbols_data[trading_instrument].index)
    correlation_history = {}  # history of correlations per currency pair
    # History of differences between projected close price deviation and actual
    # close price deviation per currency pair
    delta_projected_actual_history = {}
    # History of differences between final projected close price deviation for
    # trading instrument and actual close price deviation
    final_delta_projected_history = []

    # Variables for trading strategy trade, position and pnl management

    # Track buy/sell orders: buy=+1, sell=-1, no action=0
    orders = []
    # Track positions: long=+ve, short=-ve, flat/no position=0
    positions = []
    # Track total p&l = closed pnl + open pnl
    pnls = []
    # Price at which last buy trade was made; used to prevent over trading
    last_buy_price = 0
    # Price at which last sell trade was made; used to prevent over trading
    last_sell_price = 0
    # Current position of the trading strategy
    position = 0
    # Sum of products of buy_trade_price and buy_trade_qty for every buy trade
    # made since last time being flat
    buy_sum_price_qty = 0
    # Summation of buy_trade_qty for every buy trade made since last time being
    # flat
    buy_sum_qty = 0
    # Sum of products of sell_trade_price and sell_trade_qty for every sell
    # trade made since last time being flat
    sell_sum_price_qty = 0
    # Sum of sell_trade_qty for every sell Trade made since last time being
    # flat
    sell_sum_qty = 0
    # Open/unrealised PnL marked to market
    open_pnl = 0
    # Closed/realised PnL so far
    closed_pnl = 0

    # Statistical arbitrage trading strategy main loop
    for i in range(num_days):
        close_prices = {}

        # Build close price series, compute SMA for each symbol and price
        # deviation from SMA for each symbol
        for symbol in symbols_data:
            close_prices[symbol] = symbols_data[symbol]['Close'].iloc[i]
            if not symbol in price_history.keys():
                price_history[symbol] = []
                price_deviation_from_sma[symbol] = []

            price_history[symbol].append(close_prices[symbol])
            # Only track at most sma_time_periods number of prices for SMA
            if len(price_history[symbol]) > sma_time_periods:
                del price_history[symbol][0]

            sma = stats.mean(price_history[symbol])  # Rolling SMA
            # Calculate price deviation from rolling SMA
            price_deviation_from_sma[symbol].append(close_prices[symbol] - sma)
            if len(price_deviation_from_sma[symbol]) > price_dev_num_prices:
                del price_deviation_from_sma[symbol][0]

        # Compute covarianace and correlation between trading instrument and
        # every other lead symbol. Also compute projected price deviation and
        # find delta between projected and actual price deviations.
        projected_dev_from_sma_using = {}
        for symbol in symbols_data:
            # No need to find relationship between trading symbol and itself
            if symbol == trading_instrument:
                continue

            correlation_label = trading_instrument + '<-' + symbol

            # Create correlation history and delta projected lists for first
            # entry for the pair in the history dictionary
            if correlation_label not in correlation_history.keys():
                correlation_history[correlation_label] = []
                delta_projected_actual_history[correlation_label] = []

            # Need at least two observations to compute covariance/correlation
            if len(price_deviation_from_sma[symbol]) < 2:
                correlation_history[correlation_label].append(0)
                delta_projected_actual_history[correlation_label].append(0)
                continue

            # Calculate correlation & covariance between currancy pairs
            corr = np.corrcoef(price_deviation_from_sma[trading_instrument],
                               price_deviation_from_sma[symbol])
            cov = np.cov(price_deviation_from_sma[trading_instrument],
                         price_deviation_from_sma[symbol])
            # Get the correlation between the 2 series
            corr_trading_instrument_lead_instrument = corr[0, 1]
            # Get the covariance between the 2 series
            cov_trading_instrument_lead_instrument = cov[0, 0] / cov[0, 1]

            correlation_history[correlation_label].append(
                corr_trading_instrument_lead_instrument)

            # Calculate projected price movement and use it to find the
            # difference between the projected movement and actual movement.
            projected_dev_from_sma_using[symbol] = \
                price_deviation_from_sma[symbol][-1] * \
                cov_trading_instrument_lead_instrument

            # delta +ve => signal says trading instrument price should have
            # moved up more than what it did.
            # delta -ve => signal says trading instrument price should have
            # moved down more than what it did.
            delta_projected_actual = \
                projected_dev_from_sma_using[symbol] - \
                price_deviation_from_sma[trading_instrument][-1]
            delta_projected_actual_history[correlation_label].append(
                delta_projected_actual)

        # Combine these individual deltas between projected and actual price
        # deviation trading instrument to get one final statistical arbitrage
        # signal value for the trading instrument that is a combination of
        # projections from all the other currancy pairs. To combine these
        # different projections, use the magnitude of the correlation between
        # the trading instrument and the other currency pairs to weigh the delta
        # between projected and actual price deviations in the trading
        # instrument as predicted by the other pairs. Finally, normalise the
        # final delta value by the sum of each individual weight (magnitude of
        # correlation) and that is what will be used as the final signal to
        # build the trading strategy around.

        # Weigh predictions from each pair; weight is the correlation between
        # those pairs. The sum of weights is the sum of correlations for each
        # symbol with the trading instrument
        sum_weights = 0
        for symbol in symbols_data:
            # No need to find relationship between trading instrument and itself
            if symbol == trading_instrument:
                continue

            correlation_label = trading_instrument + '<-' + symbol
            sum_weights += abs(correlation_history[correlation_label][-1])

        # Final prediction of price deviation in trading instrument, weighing
        # projections from all other symbols.
        final_delta_projected = 0
        close_price = close_prices[trading_instrument]
        for symbol in symbols_data:
            # No need to find relationship between trading instrument and itself
            if symbol == trading_instrument:
                continue

            correlation_label = trading_instrument + '<-' + symbol

            # Weigh projection from a symbol by correlation
            final_delta_projected += \
                abs(correlation_history[correlation_label][-1]) \
                * delta_projected_actual_history[correlation_label][-1]

        # Normalise by dividing by sum of weights for all pairs
        if sum_weights != 0:
            final_delta_projected /= sum_weights
        else:
            final_delta_projected = 0

        final_delta_projected_history.append(final_delta_projected)

        # Check trading signal against trading parameters/thresholds and
        # positions to trade.

        # Performa sell trade at close_prices on the following conditions:
        # 1. Statistical arbitrage trading signal value is below sell entry
        #    threshold and the difference between last trade price and current
        #    price is different enough.
        # 2. We are long (+ve position) and current position is profitable
        #    enough to lock profit.
        if ((final_delta_projected < stat_arb_value_for_sell_entry and abs(
                close_price - last_sell_price) > min_price_move_from_last_trade)
                or
                (position > 0 and (open_pnl > min_profit_to_close))):
            orders.append(-1)  # mark the sell trade
            last_sell_price = close_price
            position -= num_shares_per_trade  # reduce position by size of trade
            sell_sum_price_qty += close_price * num_shares_per_trade
            sell_sum_qty += num_shares_per_trade
            print('Sell ', num_shares_per_trade, ' @ ', close_price,
                  'Position: ', position)
            print('OpenPnL: ', open_pnl, ' ClosedPnL: ', closed_pnl,
                  ' TotalPnL: ', open_pnl + closed_pnl)

        # Perform a buy trade at close_prices on the following conditions:
        # 1. Statistical arbitrage trading signal value is above buy entry
        #    threshold and the difference between last trade price and current
        #    price is different enough.
        # 2. We are short (-ve position) and current position is profitable
        #    enough to lock profit.
        elif ((final_delta_projected > stat_arb_value_for_buy_entry and abs(
                close_price - last_buy_price) > min_price_move_from_last_trade)
              or
              (position < 0 and (open_pnl > min_profit_to_close))):
            orders.append(+1)  # mark the buy trade
            last_buy_price = close_price
            position += num_shares_per_trade  # increase position by trade size
            buy_sum_price_qty += close_price * num_shares_per_trade
            buy_sum_qty += num_shares_per_trade
            print('Buy ', num_shares_per_trade, ' @ ', close_price,
                  'Position: ', position)
            print('OpenPnL: ', open_pnl, ' ClosedPnL: ', closed_pnl,
                  ' TotalPnL: ', open_pnl + closed_pnl)
        else:
            # No trade since none of the conditions were met to buy or sell
            orders.append(0)

        positions.append(position)

        # Update open/unrealised and closed/realised positions
        open_pnl = 0
        if position > 0:
            if sell_sum_qty > 0:
                # Long position and some sell trades have been made against it,
                # clsoe that amount based on how much was sold against this
                # long position.
                open_pnl = abs(sell_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark the remaining position to marked i.e. pnl would be what it
            # would be if we cosed at current price
            open_pnl += abs(sell_sum_qty - position) * (
                    close_price - buy_sum_price_qty / buy_sum_qty)
        elif position < 0:
            if buy_sum_qty > 0:
                # Short position and some buy trades have been made against it,
                # close that amount based on how much was bought against this
                # short position.
                open_pnl = abs(buy_sum_qty) * (
                        sell_sum_price_qty / sell_sum_qty
                        - buy_sum_price_qty / buy_sum_qty)
            # Mark the remaining position to market i.e. pnl would be what it
            # would if we cosed at current price.
            open_pnl += abs(buy_sum_qty - position) * (
                    sell_sum_price_qty / sell_sum_qty - close_price)
        else:
            # Flat, so update closed_pnl and reset tracking variables for
            # position and pnls.
            closed_pnl += sell_sum_price_qty - buy_sum_price_qty
            buy_sum_price_qty = 0
            buy_sum_qty = 0
            sell_sum_price_qty = 0
            sell_sum_qty = 0
            last_buy_price = 0
            last_sell_price = 0

        pnls.append(closed_pnl + open_pnl)

    # Plot correlations between trading instrument and other currency pairs
    cycol = cycle('bgrcmky')
    plt.figure()
    plt.title('Correlation Between Trading Instrument & Other Currency Pairs')
    correlation_data = pd.DataFrame()
    for symbol in symbols_data:
        if symbol == trading_instrument:
            continue

        correlation_label = trading_instrument + '<-' + symbol
        correlation_data = correlation_data.assign(
            label=pd.Series(correlation_history[correlation_label],
                            index=symbols_data[symbol].index))
        ax = correlation_data['label'].plot(color=next(cycol), lw=2.0,
                                            label='Correlation '
                                                  + correlation_label)

    for i in np.arange(-1, 1, 0.25):
        plt.axhline(y=i, lw=0.5, color='k')
    plt.legend()
    plt.grid()

    # Plot statistical arbitrage signal provided by each currency pair
    plt.figure()
    plt.title('Statistical Arbitrage Signal by Currency Pair')
    delta_projected_actual_data = pd.DataFrame()
    for symbol in symbols_data:
        if symbol == trading_instrument:
            continue

        projection_label = trading_instrument + '<-' + symbol
        delta_projected_actual_data = delta_projected_actual_data.assign(
            StatArbTradingSignal=pd.Series(
                delta_projected_actual_history[projection_label],
                index=symbols_data[trading_instrument].index))
        ax = delta_projected_actual_data['StatArbTradingSignal'].plot(
            color=next(cycol), lw=1.0, label='StatArbTradingSignal '
                                             + projection_label)
    plt.legend()
    plt.grid()

    # Prepare DataFrame from the trading strategy results
    delta_projected_actual_data = delta_projected_actual_data.assign(
        ClosePrice=pd.Series(symbols_data[trading_instrument]['Close'],
                             index=symbols_data[trading_instrument].index))
    delta_projected_actual_data = delta_projected_actual_data.assign(
        FinalStatArbTradingSignal=pd.Series(
            final_delta_projected_history,
            index=symbols_data[trading_instrument].index))
    delta_projected_actual_data = delta_projected_actual_data.assign(
        Trades=pd.Series(orders, index=symbols_data[trading_instrument].index))
    delta_projected_actual_data = delta_projected_actual_data.assign(
        Position=pd.Series(positions,
                           index=symbols_data[trading_instrument].index))
    delta_projected_actual_data = delta_projected_actual_data.assign(
        PnL=pd.Series(pnls, index=symbols_data[trading_instrument].index))
    return delta_projected_actual_data
