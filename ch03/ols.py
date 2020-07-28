"""Ordinary Least Squares Regression."""
from sklearn import linear_model

import algolib.finml as finml
from algolib.data import get_google_data


def main():
    goog_data = get_google_data('data/goog_data_large.pkl',
                                start_date='2001-01-01',
                                end_date='2018-01-01')
    goog_data, x, y = finml.create_regression_trading_condition(goog_data)
    x_train, x_test, y_train, y_test = finml.create_train_split_group(x, y)

    ols = linear_model.LinearRegression()
    ols.fit(x_train, y_train)

    print('Coefficients:\n', ols.coef_)


if __name__ == '__main__':
    main()
