"""Regularisation and shrinkage - Ridge Regression
This module attempts to improve on the OLS predictions by using regularisation
and coefficient shrinkage using Ridge Regression. Regularisation is a technique
that introduces a penalty term on the coefficient weights and making that part
of the mean square error, which regression tries to minimise. This lets
coefficient values grow, but only if there is a comparable decrease in MSE
values. Conversely, if reducing the coefficient weights does not increase the
MSE values too much, then it will shrink those coefficients. The extra penalty
term is known as the regularisation term, and since it results in a reduction
of the magnitudes of coefficients, it is known as shrinkage.

Depending on the type of penalty term involving magnitudes of coefficients, it
is either L1 or L2 regularisation. When the penalty term is the sum of the 
absolute values of all coefficients, this is known as L1 regularisation (LASSO),
and when the penalty term is the sum of the squared values of the coefficients,
this is known as L2 regularisation (Ridge). It is possible to combine both L1
and L2 regularisation in a technique called elastic net regression.

The amount of penalty added is controlled by tuning the regularisation 
hyperparameter. In the case of elastic net regression, there are two 
regularisation hyperparameters, one for L1 penalty and the other for L2 penalty.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import algolib.data as data
import algolib.finml as finml


def main():
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)
    goog_data = data.get_google_data('data/goog_data_large.pkl',
                                     start_date='2001-01-01',
                                     end_date='2018-01-01')
    # Features are open - close price and high - low price
    # Target is difference in daily close price
    goog_data, x, y = finml.create_regression_trading_condition(goog_data)

    # Split into train (80%) and test data sets
    x_train, x_test, y_train, y_test = finml.create_train_split_group(x, y)

    # Ridge regression model with regularisation parameter 0.1
    ridge = linear_model.Ridge(alpha=10000)
    ridge.fit(x_train, y_train)
    print('Coefficients:\n', ridge.coef_)

    # Evaluate model on training data
    print(f'Mean squared error on training data: '
          f'{mean_squared_error(y_train, ridge.predict(x_train))}')
    # Explained variance score: 1 is perfect prediction
    print(f'Variance score on training data: '
          f'{r2_score(y_train, ridge.predict(x_train))}')
    # Evaluate model on test data
    print(f'Mean squared error on test data: '
          f'{mean_squared_error(y_test, ridge.predict(x_test))}')
    # Explained variance score: 1 is perfect prediction
    print(f'Variance score on test data: '
          f'{r2_score(y_test, ridge.predict(x_test))}')

    # LASSO regression model to predict prices and calculate strategy returns
    goog_data['Predicted_signal'] = ridge.predict(x)
    goog_data['GOOG_Returns'] = np.log(
        goog_data['Close'] / goog_data['Close'].shift(1))
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


if __name__ == "__main__":
    main()
