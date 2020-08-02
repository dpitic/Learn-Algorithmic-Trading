"""Support Vector Machine.
This module implements a trading signal using Support Vector Machine (SVM) for
classification. The principle of this algorithm is to find a hyperplane that
separates the data into two classes.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

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
    goog_data, x, y = finml.create_classification_trading_condition(goog_data)

    # Split into train (80%) and test data sets
    x_train, x_test, y_train, y_test = finml.create_train_split_group(x, y)

    # SVM classification model
    svm = SVC()
    y_train_array = y_train.to_numpy().ravel()
    svm.fit(x_train, y_train_array)
    accuracy_train = accuracy_score(y_train, svm.predict(x_train))
    print('Training accuracy:', accuracy_train)
    accuracy_test = accuracy_score(y_test, svm.predict(x_test))
    print('Test accuracy:', accuracy_test)

    # Predict whether the price goes up or down
    goog_data['Predicted_signal'] = svm.predict(x)
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
