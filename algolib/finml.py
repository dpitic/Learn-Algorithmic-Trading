"""Financial Machine Learning."""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def create_classification_trading_condition(df):
    """Return classification problem input and target variables.

    The classification response variable is +1 if the future close price is
    higher than the current close prise, and -1 if the future close price is
    lower than the current close price. This function assumes the future close
    price is not the same as the current close price, i.e. only two categorical
    values are supported. It appends the two features and the target columns to
    the input DataFrame, which it returns along with the features DataFrame and
    Target Series.
    :param DataFrame df: Trading data.
    :return: Original trading DataFrame with features and target columns
        appended, along with features DataFrame and target Series.
    """
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High = df.Low
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
    df = df.dropna()
    x = df[['Open-Close', 'High-Low']]
    y = df[['Target']]
    return df, x, y


def create_regression_trading_condition(df):
    """Return regression problem input and target variables.

    The two features are: open - close price and high - low price. The target
    is the difference in daily close price. The regression response variable is
    a positive value if the price increases in the future, negative value if
    the price decreases in the future, and zero if the price does not change.
    The sign of the value indicates the direction and the magnitude of the
    response variable captures the magnitude of the price move.

    This function appends the two features and the target columns to the input
    DataFrame, which it returns along with the features DataFrame and target
    Series.
    :param DataFrame df: Trading data.
    :return: Original trading DataFrame with features and target columns
        appended, along with features DataFrame and target Series.
    """
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    df['Target'] = df['Close'].shift(-1) - df['Close']
    df = df.dropna()
    x = df[['Open-Close', 'High-Low']]
    y = df[['Target']]
    return df, x, y


def create_train_split_group(x, y, train_ratio=0.8):
    """Return train and test split data sets.

    This function splits the input and target data sets using the specified 
    train data set size, and without shuffling the data sets.

    :param x: Input data set.
    :param y: Target data set.
    :param int train_ratio: Train data set size, default=0.8
    :return: x_train, x_test, y_train, y_test
    """
    return train_test_split(x, y, shuffle=False, train_size=train_ratio)


def calculate_return(df, split_value, symbol):
    """Return the cumulative return of the specified symbol.

    This function calculates the cumulative sum of the symbol from the starting
    point specified by split_value. It calculates the strategy returns as the
    product of the symbol returns and the predicted signal, which it inserts in
    the provided DataFrame.
    :param DataFrame df: Trading data.
    :param int split_value: Start index for cumulative sum calculation.
    :param str symbol: Trading symbol used for calculation.
    :return: Series with cumulative returns and the input DataFrame has a new
        'Strategy_Returns' column appended which is the product of the symbol
        returns and the predicted signal.
    """
    cum_return = df[split_value:][f'{symbol}_Returns'].cumsum() * 100
    df['Strategy_Returns'] = df[f'{symbol}_Returns'] * \
                             df['Predicted_signal'].shift(1)
    return cum_return


def calculate_strategy_return(df, split_value):
    """Return cumulative strategy returns.

    Calculate the cumulative sum of the strategy returns starting from the
    index specified by split_value.
    :param DataFrame df: Trading data.
    :param int split_value: Start index for cumulative sum calculation.
    :return: Series of cumulative sum strategy return.
    """
    cum_strategy_return = df[split_value:]['Strategy_Returns'].cumsum() * 100
    return cum_strategy_return


def plot_chart(cum_symbol_return, cum_strategy_return, symbol):
    """Plot the cumulative symbol returns and the cumulative strategy returns.

    This function creates a new figure and expects the caller to call show().
    """
    plt.figure(figsize=(10, 5))
    plt.plot(cum_symbol_return, label=f'{symbol}_Returns')
    plt.plot(cum_strategy_return, label='Strategy Returns')
    plt.legend()
    plt.grid()


def sharpe_ratio(symbol_returns, strategy_returns):
    """Return the Sharpe Ratio."""
    strategy_std = strategy_returns.std()
    sharpe = (strategy_returns - symbol_returns) / strategy_std
    return sharpe.mean()
