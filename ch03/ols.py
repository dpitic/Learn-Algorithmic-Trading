"""Ordinary Least Squares Regression."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import algolib.finml as finml
from algolib.data import get_google_data


def calculate_return(df, split_value, symbol):
    cum_goog_return = df[split_value:][f'{symbol}_Returns'].cumsum() * 100
    df['Strategy_Returns'] = df[f'{symbol}_Returns'] * \
                             df['Predicted_signal'].shift(1)
    return cum_goog_return


def calculate_strategy_return(df, split_value, symbol):
    cum_strategy_return = df[split_value:]['Strategy_Returns'].cumsum() * 100
    return cum_strategy_return


def plot_shart(cum_symbol_return, cum_strategy_return, symbol):
    plt.figure(figsize=(10, 5))
    plt.plot(cum_symbol_return, label=f'{symbol}_Returns')
    plt.plot(cum_strategy_return, label='Strategy Returns')
    plt.legend()
    plt.grid()


def sharpe_ratio(symbol_returns, strategy_returns):
    strategy_std = strategy_returns.std()
    sharpe = (strategy_returns - symbol_returns) / strategy_std
    return sharpe.mean()


def main():
    goog_data = get_google_data('data/goog_data_large.pkl',
                                start_date='2001-01-01',
                                end_date='2018-01-01')
    # Features are open - close price and high - low price
    # Target is difference in daily close price
    goog_data, x, y = finml.create_regression_trading_condition(goog_data)
    # Visualise the data
    pd.plotting.scatter_matrix(goog_data[['Open-Close', 'High-Low', 'Target']],
                               grid=True, diagonal='kde', alpha=0.5)

    # Split into train (80%) and test data sets
    x_train, x_test, y_train, y_test = finml.create_train_split_group(x, y)

    # Ordinary least squares model
    ols = linear_model.LinearRegression()
    ols.fit(x_train, y_train)
    # Optimal weights for the two features
    print('Coefficients:\n', ols.coef_)

    # Evaluate model on train data
    print(f'Mean squared error on training data: '
          f'{mean_squared_error(y_train, ols.predict(x_train))}')
    # Explained variance score: 1 is perfect prediction
    print(f'Variance score on training data: '
          f'{r2_score(y_train, ols.predict(x_train))}')
    # Evaluate model on test data
    print(f'Mean squared error on test data: '
          f'{mean_squared_error(y_test, ols.predict(x_test))}')
    # Explained variance score: 1 is perfect prediction
    print(f'Variance score on test data: '
          f'{r2_score(y_test, ols.predict(x_test))}')

    # Use OLS regression model to predict prices and calculate strategy returns
    goog_data['Predicted_signal'] = ols.predict(x)
    goog_data['GOOG_Returns'] = np.log(
        goog_data['Close'] / goog_data['Close'].shift(1))
    print(goog_data)

    cum_goog_return = calculate_return(goog_data, split_value=len(x_train),
                                       symbol='GOOG')
    cum_strategy_return = calculate_strategy_return(goog_data,
                                                    split_value=len(x_train),
                                                    symbol='GOOG')

    plot_shart(cum_goog_return, cum_strategy_return, symbol='GOOG')

    print(f'Sharpe ratio: {sharpe_ratio(cum_strategy_return, cum_goog_return)}')

    plt.show()


if __name__ == '__main__':
    main()
