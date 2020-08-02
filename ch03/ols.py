"""Ordinary Least Squares Regression.
This module uses two features: open - close price and high - low price to
predict returns using Ordinary Least Squares (OLS) Regression techniques. The
target variable is the difference in daily close price.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import algolib.finml as finml
from algolib.data import get_google_data


def main():
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)
    goog_data = get_google_data('data/goog_data_large.pkl',
                                start_date='2001-01-01',
                                end_date='2018-01-01')
    # Features are open - close price and high - low price
    # Target is difference in daily close price
    goog_data, x, y = finml.create_regression_trading_condition(goog_data)

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
    goog_data['GOOG_Returns'] = \
        np.log(goog_data['Close'] / goog_data['Close'].shift(1))
    print(goog_data)

    # Calculate cumulative returns for the test data (train data onwards)
    cum_goog_return = \
        finml.calculate_return(goog_data, split_value=len(x_train),
                               symbol='GOOG')
    cum_strategy_return = \
        finml.calculate_strategy_return(goog_data, split_value=len(x_train))

    finml.plot_chart(cum_goog_return, cum_strategy_return, symbol='GOOG')

    sharpe = finml.sharpe_ratio(cum_strategy_return, cum_goog_return)
    print(f'Sharpe ratio: {sharpe}')

    # Display plots and block
    plt.show()


if __name__ == '__main__':
    main()
