"""Financial Machine Learning."""
import numpy as np
from sklearn.model_selection import train_test_split


def create_classification_trading_condition(df):
    """Return classification problem input and target variables.

    The classification response variable is +1 if the future close price is
    higher than the current close prise, and -1 if the future close price is
    lower than the current close price. This function assumes the future close
    price is not the same as the current close price, i.e. only two categorical
    values are supported.
    """
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High = df.Low
    df = df.dropna()
    x = df[['Open-Close', 'High-Low']]
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
    return x, y


def create_regression_trading_condition(df):
    """Return regression problem input and target variables.

    The regression response variable is a positive value if the price increases
    in the future, negative value if the price decreases in the future, and
    zero if the price does not change. The sign of the value indicates the
    direction and the magnitude of the response variable captures the magnitude
    of the price move.
    """
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    df = df.dropna()
    x = df[['Open-Close', 'High-Low']]
    y = df['Close'].shift(-1) - df['Close']
    return x, y


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
